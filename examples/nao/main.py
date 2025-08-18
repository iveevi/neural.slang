from ..ngp import AddressBasedMLP, MLP, Adam, Optimizer, DenseGrid
from ..util import *
from PIL import Image
from common import *
from dataclasses import dataclass
from typing import Any
import numpy as np
import slangpy as spy
import trimesh


HERE = ROOT / "examples" / "nao"


@dataclass
class Cylinder:
    center: spy.float3
    radius: float
    height: float

    def dict(self):
        return {
            "center": self.center,
            "radius": self.radius,
            "height": self.height,
        }


class RenderingPipeline:
    @staticmethod
    def load_specialization_module(device: spy.Device, mlp: MLP):
        source = f"""
        export static const int Hidden = {mlp.hidden};
        export static const int HiddenLayers = {mlp.hidden_layers};
        """
        return device.load_module_from_source("specialization", source)

    def __init__(self, device: spy.Device, mlp: MLP):
        SOURCE = HERE / "slang" / "main.slang"
        
        self.device = device
        self.module = device.load_module(str(SOURCE))
        self.specialization_module = self.load_specialization_module(device, mlp)
        
        self.network_type = self.module.layout.find_type_by_name("network")
        print("network_type", self.network_type)
       
        # Neural rendering pipeline
        # self.render_neural_pipeline = create_compute_pipeline(device, self.module, [self.specialization_module], "render_neural")
        
        # Backward pass pipeline
        # self.backward_pipeline = create_compute_pipeline(device, self.module, [self.specialization_module], "backward")

        # Optimization pipeline
        self.update_mlp_pipeline = create_compute_pipeline(device, self.module, [self.specialization_module], "update_mlp")
        self.update_grid_pipeline = create_compute_pipeline(device, self.module, [self.specialization_module], "update_grid")

        # Reference pipeline
        self.reference_pipeline = self.create_rasterization_pipeline(device, "reference_fragment")

        # Neural shading pipeline
        self.neural_shading_pipeline = self.create_rasterization_pipeline(device, "render_neural_shading")

        # Backward pass pipeline
        self.backward_pipeline = self.create_rasterization_pipeline(device, "render_backward")

    def create_rasterization_pipeline(self, device: spy.Device, fragment_shader: str):
        program = device.link_program(
            modules=[self.module, self.specialization_module],
            entry_points=[
                self.module.entry_point("reference_vertex"),
                self.module.entry_point(fragment_shader),
            ],
        )
        
        input_layout = device.create_input_layout(
            input_elements=[
                {
                    "semantic_name": "POSITION",
                    "semantic_index": 0,
                    "format": spy.Format.rgb32_float,
                    "offset": 0,
                    "buffer_slot_index": 0,
                },
                {
                    "semantic_name": "NORMAL",
                    "semantic_index": 0,
                    "format": spy.Format.rgb32_float,
                    "offset": 0,
                    "buffer_slot_index": 1,
                },
                {
                    "semantic_name": "TEXCOORD",
                    "semantic_index": 0,
                    "format": spy.Format.rg32_float,
                    "offset": 0,
                    "buffer_slot_index": 2,
                }
            ],
            vertex_streams=[
                { "stride": 4 * 3 },
                { "stride": 4 * 3 },
                { "stride": 4 * 2 }
            ]
        )
        
        return device.create_render_pipeline(
            program=program,
            input_layout=input_layout,
            primitive_topology=spy.PrimitiveTopology.triangle_list,
            targets=[
                { "format": spy.Format.rgba32_float },
            ],
            depth_stencil={
                "format": spy.Format.d32_float,
                "depth_test_enable": True,
                "depth_write_enable": True,
                "depth_func": spy.ComparisonFunc.less,
            }
        )

    @staticmethod
    def create_render_pass_args(target: tuple[spy.Texture, spy.TextureView], depth_target: spy.TextureView):
        return {
            "color_attachments": [
                {
                    "view": target[1],
                    "clear_value": [0.0, 0.0, 0.0, 1.0],
                    "load_op": spy.LoadOp.clear,
                    "store_op": spy.StoreOp.store,
                }
            ],
            "depth_stencil_attachment": {
                "view": depth_target,
                "depth_clear_value": 1.0,
                "depth_load_op": spy.LoadOp.clear,
                "depth_store_op": spy.StoreOp.store,
            }
        }

    @staticmethod
    def create_render_state(mesh: TriangleMesh, target: tuple[spy.Texture, spy.TextureView]):
        state = spy.RenderState()
        state.vertex_buffers = [mesh.vertices, mesh.normals, mesh.uvs]
        state.index_buffer = mesh.triangles
        state.index_format = spy.IndexFormat.uint32
        state.viewports = [spy.Viewport.from_size(target[0].width, target[0].height)]
        state.scissor_rects = [spy.ScissorRect.from_size(target[0].width, target[0].height)]
        return state

    def render_reference(self,
                         camera: Camera,
                         mesh: TriangleMesh,
                         target: tuple[spy.Texture, spy.TextureView],
                         depth_target: spy.TextureView,
                         diffuse_texture: spy.Texture,
                         sampler: spy.Sampler):
        render_pass_args: Any = self.create_render_pass_args(target, depth_target)
        
        command_encoder = self.device.create_command_encoder()
        
        with command_encoder.begin_render_pass(render_pass_args) as cmd:
            shader_object = cmd.bind_pipeline(self.reference_pipeline)
            
            cursor = spy.ShaderCursor(shader_object)
            cursor.view = camera.view_matrix()
            cursor.perspective = camera.perspective_matrix()
            cursor.diffuseTexture = diffuse_texture
            cursor.diffuseSampler = sampler
            
            state = self.create_render_state(mesh, target)
            cmd.set_render_state(state)
            
            params = spy.DrawArguments()
            params.instance_count = 1
            params.vertex_count = mesh.triangle_count * 3
            cmd.draw_indexed(params)
            
        self.device.submit_command_buffer(command_encoder.finish())

    def render_neural_shading(self,
                              camera: Camera,
                              mesh: TriangleMesh,
                              mlp: MLP,
                              grid: DenseGrid,
                              target: tuple[spy.Texture, spy.TextureView],
                              depth_target: spy.TextureView):
        render_pass_args: Any = self.create_render_pass_args(target, depth_target)
        command_encoder = self.device.create_command_encoder()
        
        with command_encoder.begin_render_pass(render_pass_args) as cmd:
            shader_object = cmd.bind_pipeline(self.neural_shading_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.view = camera.view_matrix()
            cursor.perspective = camera.perspective_matrix()
            cursor.mlp = mlp.dict()
            cursor.grid = grid.dict()
            state = self.create_render_state(mesh, target)
            cmd.set_render_state(state)
            params = spy.DrawArguments()
            params.instance_count = 1
            params.vertex_count = mesh.triangle_count * 3
            cmd.draw_indexed(params)
            
        self.device.submit_command_buffer(command_encoder.finish())

    def render_backward(self,
                        camera: Camera,
                        mesh: TriangleMesh,
                        reference_texture: spy.Texture,
                        reference_sampler: spy.Sampler,
                        mlp: MLP,
                        grid: DenseGrid,
                        target: tuple[spy.Texture, spy.TextureView],
                        depth_target: spy.TextureView,
                        loss_buffer: spy.Buffer,
                        valid_buffer: spy.Buffer,
                        boost: float):
        render_pass_args: Any = self.create_render_pass_args(target, depth_target)
        command_encoder = self.device.create_command_encoder()
        
        with command_encoder.begin_render_pass(render_pass_args) as cmd:
            shader_object = cmd.bind_pipeline(self.backward_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.view = camera.view_matrix()
            cursor.perspective = camera.perspective_matrix()
            cursor.referenceTexture = reference_texture
            cursor.referenceSampler = reference_sampler
            cursor.mlp = mlp.dict()
            cursor.grid = grid.dict()
            cursor.targetResolution = (target[0].width, target[0].height)
            cursor.boost = boost
            cursor.lossBuffer = loss_buffer
            cursor.validBuffer = valid_buffer
            state = self.create_render_state(mesh, target)
            cmd.set_render_state(state)
            params = spy.DrawArguments()
            params.instance_count = 1
            params.vertex_count = mesh.triangle_count * 3
            cmd.draw_indexed(params)
            
        self.device.submit_command_buffer(command_encoder.finish())

    def update_mlp(self, mlp: MLP, optimizer: Optimizer, optimizer_states: spy.Buffer):
        command_encoder = self.device.create_command_encoder()

        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.update_mlp_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.mlp = mlp.dict()
            cursor.optimizer = optimizer.dict()
            cursor.optimizerStates = optimizer_states
            cmd.dispatch(thread_count=(mlp.parameter_count, 1, 1))

        self.device.submit_command_buffer(command_encoder.finish())

    def update_grid(self, grid: DenseGrid, optimizer: Optimizer, optimizer_states: spy.Buffer):
        command_encoder = self.device.create_command_encoder()
        
        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.update_grid_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.grid = grid.dict()
            cursor.optimizer = optimizer.dict()
            cursor.optimizerStates = optimizer_states
            cmd.dispatch(thread_count=(grid.parameter_count, 1, 1))
            
        self.device.submit_command_buffer(command_encoder.finish())


def alloc_target_texture(device: spy.Device, width: int, height: int):
    texture = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba32_float,
        width=width,
        height=height,
        usage=spy.TextureUsage.shader_resource
            | spy.TextureUsage.unordered_access
            | spy.TextureUsage.render_target,
    )
    view = texture.create_view()
    return texture, view


def main():
    device = create_device()

    optimizer = Adam(alpha=1e-2)
    mlp = AddressBasedMLP.new(device, hidden=32, hidden_layers=2, input=3, output=3)
    grid = DenseGrid.new(device, dimension=3, features=3, resolution=128)
    mlp_optimizer_states = mlp.alloc_optimizer_states(device, optimizer)
    grid_optimizer_states = grid.alloc_optimizer_states(device, optimizer)
    
    rendering_pipeline = RenderingPipeline(device, mlp)
    
    assert device.has_feature(spy.Feature.rasterization)
    
    geometry = trimesh.load_mesh(ROOT / "resources" / "spinosa.obj")
    print("geometry", geometry)
    print("geometry.visual", geometry.visual)
    
    # Calculate mesh properties for camera positioning
    mesh_center = geometry.centroid.astype(np.float32)
    mesh_bounds = geometry.bounds
    print("mesh_bounds", mesh_bounds)
    mesh_size = np.linalg.norm(mesh_bounds[1] - mesh_bounds[0])
    
    # Calculate bounding cylinder (vertical orientation - Y axis is height)
    vertices = geometry.vertices.astype(np.float32)
    relative_vertices = vertices - mesh_center
    cylinder_radius = np.max(np.sqrt(relative_vertices[:, 0] ** 2 + relative_vertices[:, 2] ** 2))
    cylinder_height = mesh_bounds[1][1] - mesh_bounds[0][1]
    
    print(f"Mesh center: {mesh_center}")
    print(f"Mesh bounds: {mesh_bounds}")
    print(f"Mesh size: {mesh_size}")
    print(f"Bounding cylinder - center: {mesh_center}, radius: {cylinder_radius:.3f}, height: {cylinder_height:.3f}")
    
    mesh = TriangleMesh.new(device, geometry)
    
    show_reference = True
    pause_orbit = False
    
    def keyboard_hook(event: spy.KeyboardEvent):
        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.tab:
                nonlocal show_reference
                show_reference = not show_reference
            if event.key == spy.KeyCode.space:
                nonlocal pause_orbit
                pause_orbit = not pause_orbit

    app = App(device, keyboard_hook=keyboard_hook)
    
    texture, texture_view = alloc_target_texture(device, 512, 512)

    ref_texture, ref_texture_view = alloc_target_texture(device, 128, 128)
    train_texture, train_texture_view = alloc_target_texture(device, 128, 128)

    depth_texture = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.d32_float,
        width=app.width,
        height=app.height,
        usage=spy.TextureUsage.depth_stencil,
        data=None,
    )

    depth_texture_view = depth_texture.create_view()
    
    # Load texture
    texture_image = Image.open(ROOT / "resources" / "spinosa_albedo.jpg")
    if texture_image.mode == 'RGB':
        texture_image = texture_image.convert('RGBA')
    texture_data = np.array(texture_image).astype(np.uint8)
    
    diffuse_texture = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba8_unorm,
        width=texture_image.width,
        height=texture_image.height,
        usage=spy.TextureUsage.shader_resource,
        data=texture_data,
    )
    
    # Create sampler
    sampler = device.create_sampler(
        min_filter=spy.TextureFilteringMode.linear,
        mag_filter=spy.TextureFilteringMode.linear,
        mip_filter=spy.TextureFilteringMode.linear,
        address_u=spy.TextureAddressingMode.wrap,
        address_v=spy.TextureAddressingMode.wrap,
        address_w=spy.TextureAddressingMode.wrap,
    )
    
    camera = Camera.new(aspect_ratio=app.width / app.height)
    
    # Position camera to view the mesh properly using calculated center and size
    camera_distance = mesh_size * 1.5  # Distance based on mesh size
    camera.transform.position = mesh_center + np.array([0.0, 0.0, camera_distance])
    
    loss_buffer = create_buffer_32b(device, np.zeros((app.width * app.height,), dtype=np.float32))
    valid_buffer = create_buffer_32b(device, np.zeros((app.width * app.height,), dtype=np.float32))
    
    history = []
    counter = 0
    bound = Cylinder(mesh_center, cylinder_radius, cylinder_height)

    def set_orbit_camera(time: float):
        orbit_radius = mesh_size * 1.2  # Orbit radius based on mesh size
        camera.transform.position = mesh_center + orbit_radius * np.array(
            (np.cos(time),
            0.2 * np.sin(3 * time),
            np.sin(time))
        )
        camera.transform.look_at(mesh_center)

    orbit_third = 2 * np.pi / 3
    
    def loop(frame: Frame):
        # Orbit time
        nonlocal counter
        time = counter * 0.01
        if not pause_orbit:
            counter += 1

        # Orbit camera
        set_orbit_camera(time)

        # Reference rendering
        rendering_pipeline.render_reference(
            camera,
            mesh,
            (ref_texture, ref_texture_view),
            depth_texture_view,
            diffuse_texture,
            sampler,
        )

        loss_numpy = np.zeros((app.width * app.height,), dtype=np.float32)
        valid_numpy = np.zeros((app.width * app.height,), dtype=np.float32)
        loss_buffer.copy_from_numpy(loss_numpy)
        valid_buffer.copy_from_numpy(valid_numpy)

        rendering_pipeline.render_backward(
            camera,
            mesh,
            ref_texture,
            sampler,
            mlp,
            grid,
            (train_texture, train_texture_view),
            depth_texture_view,
            loss_buffer,
            valid_buffer,
            boost=(1.0 / (train_texture.width * train_texture.height))
        )

        loss = loss_buffer.to_numpy().view(np.float32)
        valid = valid_buffer.to_numpy().view(np.float32)
        loss = loss[valid == 1.0].mean()
        history.append(loss)
        
        # Rendering
        if show_reference:
            rendering_pipeline.render_reference(
                camera,
                mesh,
                (texture, texture_view),
                depth_texture_view,
                diffuse_texture,
                sampler,
            )
        else:
            rendering_pipeline.render_neural_shading(
                camera,
                mesh,
                mlp,
                grid,
                (texture, texture_view),
                depth_texture_view,
            )

        # Optimize
        rendering_pipeline.update_mlp(mlp, optimizer, mlp_optimizer_states)
        rendering_pipeline.update_grid(grid, optimizer, grid_optimizer_states)
        
        # Display
        frame.blit(texture)
    
    app.run(loop)
    
    # Plot loss
    # TODO: util method
    import seaborn as sns
    from scipy.ndimage import gaussian_filter
    import matplotlib.pyplot as plt
    sns.set_theme()
    sns.set_palette("pastel")
    sns.lineplot(history, alpha=0.5, color="green")
    sns.lineplot(gaussian_filter(history, 5), linewidth=2.5, color="green")
    plt.yscale("log")
    plt.show()

if __name__ == "__main__":
    main()