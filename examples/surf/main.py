from ..ngp import AddressBasedMLP, MLP, Adam, Optimizer, DenseGrid
from ..util import *
from PIL import Image
from common import *
from dataclasses import dataclass
from typing import Any
import numpy as np
import slangpy as spy
import trimesh


HERE = ROOT / "examples" / "surf"


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
        self.render_neural_pipeline = create_compute_pipeline(device, self.module, [self.specialization_module], "render_neural")
        
        # Backward pass pipeline
        self.backward_pipeline = create_compute_pipeline(device, self.module, [self.specialization_module], "backward")

        # Optimization pipeline
        self.update_mlp_pipeline = create_compute_pipeline(device, self.module, [self.specialization_module], "update_mlp")
        self.update_grid1_pipeline = create_compute_pipeline(device, self.module, [self.specialization_module], "update_grid1")
        self.update_grid2_pipeline = create_compute_pipeline(device, self.module, [self.specialization_module], "update_grid2")

        # Reference pipeline
        reference_program = device.link_program(
            modules=[self.module, self.specialization_module],
            entry_points=[
                self.module.entry_point("reference_vertex"),
                self.module.entry_point("reference_fragment"),
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
        
        self.reference_pipeline = device.create_render_pipeline(
            program=reference_program,
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

    def render_neural(self, mlp: MLP, grid1: DenseGrid, grid2: DenseGrid, rayframe: RayFrame, target_texture: spy.Texture, cylinder: Cylinder, show_half: bool):
        command_encoder = self.device.create_command_encoder()

        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.render_neural_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.mlp = mlp.dict()
            cursor.grid1 = grid1.dict()
            cursor.grid2 = grid2.dict()
            cursor.rayFrame = rayframe.dict()
            cursor.targetTexture = target_texture
            cursor.targetResolution = (target_texture.width, target_texture.height)
            cursor.cylinder = cylinder.dict()
            cursor.showHalf = show_half
            cmd.dispatch(thread_count=(target_texture.width, target_texture.height, 1))
            
        self.device.submit_command_buffer(command_encoder.finish())
        
    def backward(self,
                 mlp: MLP,
                 grid1: DenseGrid,
                 grid2: DenseGrid,
                 rayframe: RayFrame,
                 target_texture: spy.Texture,
                 cylinder: Cylinder,
                 loss_buffer: spy.Buffer,
                 boost: float):
        command_encoder = self.device.create_command_encoder()
        
        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.backward_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.mlp = mlp.dict()
            cursor.grid1 = grid1.dict()
            cursor.grid2 = grid2.dict()
            cursor.rayFrame = rayframe.dict()
            cursor.targetTexture = target_texture
            cursor.targetResolution = (target_texture.width, target_texture.height)
            cursor.lossBuffer = loss_buffer
            cursor.cylinder = cylinder.dict()
            cursor.boost = boost
            cmd.dispatch(thread_count=(target_texture.width, target_texture.height, 1))
            
        self.device.submit_command_buffer(command_encoder.finish())
        
    def render_reference(self,
                         camera: Camera,
                         mesh: TriangleMesh,
                         target: tuple[spy.Texture, spy.TextureView],
                         depth_target: spy.TextureView,
                         diffuse_texture: spy.Texture,
                         sampler: spy.Sampler):
        render_pass_args: Any = {
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
        
        command_encoder = self.device.create_command_encoder()
        
        with command_encoder.begin_render_pass(render_pass_args) as cmd:
            shader_object = cmd.bind_pipeline(self.reference_pipeline)
            
            cursor = spy.ShaderCursor(shader_object)
            cursor.view = camera.view_matrix()
            cursor.perspective = camera.perspective_matrix()
            cursor.diffuseTexture = diffuse_texture
            cursor.diffuseSampler = sampler
            
            state = spy.RenderState()
            state.vertex_buffers = [ mesh.vertices, mesh.normals, mesh.uvs ]
            state.index_buffer = mesh.triangles
            state.index_format = spy.IndexFormat.uint32
            state.viewports = [
                spy.Viewport.from_size(target[0].width, target[0].height)
            ]
            state.scissor_rects = [
                spy.ScissorRect.from_size(target[0].width, target[0].height)
            ]
            cmd.set_render_state(state)
            
            params = spy.DrawArguments()
            params.instance_count = 1
            params.vertex_count = mesh.triangle_count * 3  # Total indices to draw
            
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

    def update_grid1(self, grid1: DenseGrid, optimizer: Optimizer, optimizer_states: spy.Buffer):
        command_encoder = self.device.create_command_encoder()
        
        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.update_grid1_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.grid1 = grid1.dict()
            cursor.optimizer = optimizer.dict()
            cursor.optimizerStates = optimizer_states
            cmd.dispatch(thread_count=(grid1.parameter_count, 1, 1))
            
        self.device.submit_command_buffer(command_encoder.finish())

    def update_grid2(self, grid2: DenseGrid, optimizer: Optimizer, optimizer_states: spy.Buffer):
        command_encoder = self.device.create_command_encoder()
        
        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.update_grid2_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.grid2 = grid2.dict()
            cursor.optimizer = optimizer.dict()
            cursor.optimizerStates = optimizer_states
            cmd.dispatch(thread_count=(grid2.parameter_count, 1, 1))
            
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
    mlp = AddressBasedMLP.new(device, hidden=16, hidden_layers=3, input=8, output=3)
    grid1 = DenseGrid.new(device, dimension=2, features=4, resolution=64)
    grid2 = DenseGrid.new(device, dimension=2, features=4, resolution=128)
    mlp_optimizer_states = mlp.alloc_optimizer_states(device, optimizer)
    grid1_optimizer_states = grid1.alloc_optimizer_states(device, optimizer)
    grid2_optimizer_states = grid2.alloc_optimizer_states(device, optimizer)
    
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
    
    # show_reference = False

    show_mode = 0
    show_reference = 0
    show_neural = 1
    show_side_by_side = 2

    pause_orbit = False
    
    def keyboard_hook(event: spy.KeyboardEvent):
        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.tab:
                nonlocal show_mode
                show_mode = (show_mode + 1) % 3
            if event.key == spy.KeyCode.space:
                nonlocal pause_orbit
                pause_orbit = not pause_orbit

    app = App(device, keyboard_hook=keyboard_hook)
    
    texture, texture_view = alloc_target_texture(device, 512, 512)
    train_texture, train_texture_view = alloc_target_texture(device, 256, 256)

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
        nonlocal counter
        time = counter * 0.01
        if not pause_orbit:
            counter += 1

        samples = 3
        shifts = np.linspace(0, 2 * np.pi, samples)
        noise = 0.1 * (2 * np.random.rand(samples) - 1) / samples
        shifts += noise

        for shift in shifts:
            set_orbit_camera(time + shift)

            # Reference rendering for training
            rendering_pipeline.render_reference(
                camera,
                mesh,
                (train_texture, train_texture_view),
                depth_texture_view,
                diffuse_texture,
                sampler,
            )

            device.wait_for_idle()
            
            # Backward pass
            boost = 1.0 / (samples * train_texture.width * train_texture.height)
            rendering_pipeline.backward(mlp, grid1, grid2, camera.rayframe(), train_texture, bound, loss_buffer, boost)
            
            device.wait_for_idle()

        # Optimize
        rendering_pipeline.update_mlp(mlp, optimizer, mlp_optimizer_states)
        rendering_pipeline.update_grid1(grid1, optimizer, grid1_optimizer_states)
        rendering_pipeline.update_grid2(grid2, optimizer, grid2_optimizer_states)
    
        loss_data = loss_buffer.to_numpy().view(np.float32)
        loss = loss_data.mean()
        history.append(loss)
        
        # Rendering
        set_orbit_camera(time)

        if show_mode == show_reference or show_mode == show_side_by_side:
            rendering_pipeline.render_reference(
                camera,
                mesh,
                (texture, texture_view),
                depth_texture_view,
                diffuse_texture,
                sampler,
            )
        if show_mode == show_neural or show_mode == show_side_by_side:
            rendering_pipeline.render_neural(
                mlp,
                grid1,
                grid2,
                camera.rayframe(),
                texture,
                bound,
                show_mode == show_side_by_side,
            )
        
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