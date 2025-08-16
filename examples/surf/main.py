from ..networks.addresses import Network, TrainingPipeline
from ..util import *
from common import *
from dataclasses import dataclass
from typing import Any
import numpy as np
import slangpy as spy
import trimesh
from PIL import Image


HERE = ROOT / "examples" / "surf"


# TODO: automatically generate slang source for training pipeline based on
# main.slang (reflection on the network paramter block)


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
    def load_specialization_module(device: spy.Device, network: Network):
        source = f"""
        export static const int Hidden = {network.hidden};
        export static const int HiddenLayers = {network.hidden_layers};
        export static const int Levels = {network.levels};
        """
        return device.load_module_from_source("specialization", source)

    def __init__(self, device: spy.Device, network: Network):
        SOURCE = HERE / "slang" / "main.slang"
        
        self.device = device
        self.module = device.load_module(str(SOURCE))
        self.specialization_module = self.load_specialization_module(device, network)
        
        self.network_type = self.module.layout.find_type_by_name("network")
        print("network_type", self.network_type)
       
        # Neural rendering pipeline
        self.render_neural_pipeline = create_compute_pipeline(device, self.module, [self.specialization_module], "render_neural")
        
        # Backward pass pipeline
        self.backward_pipeline = create_compute_pipeline(device, self.module, [self.specialization_module], "backward")

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
                { "format": spy.Format.rgba8_unorm },
            ],
            depth_stencil={
                "format": spy.Format.d32_float,
                "depth_test_enable": True,
                "depth_write_enable": True,
                "depth_func": spy.ComparisonFunc.less,
            }
        )

    def render_neural(self, network: Network, rayframe: RayFrame, target_texture: spy.Texture, cylinder: Cylinder):
        command_encoder = self.device.create_command_encoder()

        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.render_neural_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.network = network.dict()
            cursor.rayFrame = rayframe.dict()
            cursor.targetTexture = target_texture
            cursor.targetResolution = (target_texture.width, target_texture.height)
            cursor.cylinder = cylinder.dict()
            cmd.dispatch(thread_count=(target_texture.width, target_texture.height, 1))
            
        self.device.submit_command_buffer(command_encoder.finish())
        
    def backward(self, network: Network, rayframe: RayFrame, target_texture: spy.Texture, cylinder: Cylinder, loss_buffer: spy.Buffer):
        command_encoder = self.device.create_command_encoder()
        
        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.backward_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.network = network.dict()
            cursor.rayFrame = rayframe.dict()
            cursor.targetTexture = target_texture
            cursor.targetResolution = (target_texture.width, target_texture.height)
            cursor.lossBuffer = loss_buffer
            cursor.cylinder = cylinder.dict()
            cmd.dispatch(thread_count=(target_texture.width, target_texture.height, 1))
            
        self.device.submit_command_buffer(command_encoder.finish())
        
    def render_reference(self, camera: Camera, mesh: TriangleMesh, target: tuple[spy.Texture, spy.TextureView], depth_target: spy.TextureView, diffuse_texture: spy.Texture, sampler: spy.Sampler):
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


def main():
    device = create_device()
    
    network = Network(
        device,
        hidden=32,
        hidden_layers=4,
        levels=0,
        input=7,
        output=3,
    )
    
    rendering_pipeline = RenderingPipeline(device, network)
    training_pipeline = TrainingPipeline(device, network)
    
    assert device.has_feature(spy.Feature.rasterization)
    
    geometry = trimesh.load_mesh(ROOT / "resources" / "spinosa.obj")
    print("geometry", geometry)
    print("geometry.visual", geometry.visual)
    
    # Calculate mesh properties for camera positioning
    mesh_center = geometry.centroid.astype(np.float32)
    mesh_bounds = geometry.bounds
    mesh_size = np.linalg.norm(mesh_bounds[1] - mesh_bounds[0])
    
    # Calculate bounding cylinder (vertical orientation - Y axis is height)
    vertices = geometry.vertices.astype(np.float32)
    relative_vertices = vertices - mesh_center
    cylinder_radius = np.max(np.sqrt(relative_vertices[:, 0]**2 + relative_vertices[:, 2]**2))
    cylinder_height = mesh_bounds[1][1] - mesh_bounds[0][1]
    
    print(f"Mesh center: {mesh_center}")
    print(f"Mesh bounds: {mesh_bounds}")
    print(f"Mesh size: {mesh_size}")
    print(f"Bounding cylinder - center: {mesh_center}, radius: {cylinder_radius:.3f}, height: {cylinder_height:.3f}")
    
    mesh = TriangleMesh.new(device, geometry)
    
    show_reference = False
    
    def keyboard_hook(event: spy.KeyboardEvent):
        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.tab:
                nonlocal show_reference
                show_reference = not show_reference
    
    app = App(device, keyboard_hook=keyboard_hook)
    
    texture = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba8_unorm,
        width=256,
        height=256,
        usage=spy.TextureUsage.shader_resource
            | spy.TextureUsage.unordered_access
            | spy.TextureUsage.render_target,
        data=None,
    )

    # TODO: multiple lower resolution target textures for backward pass
    
    depth_texture = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.d32_float,
        width=app.width,
        height=app.height,
        usage=spy.TextureUsage.depth_stencil,
        data=None,
    )
    
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
    
    history: list[float] = []

    # TODO: toggle with 'tab' key

    texture_view = texture.create_view()
    frame_views = dict()

    bound = Cylinder(mesh_center, cylinder_radius, cylinder_height)
    
    def loop(frame: Frame):
        frame_view = None
        if frame.image not in frame_views:
            frame_view = frame.image.create_view()
            frame_views[id(frame.image)] = frame_view
        else:
            frame_view = frame_views[id(frame.image)]

        time = frame.count[0] * 0.01
        
        # Orbit around the calculated mesh centroid
        orbit_radius = mesh_size * 1.2  # Orbit radius based on mesh size
        camera.transform.position = mesh_center + orbit_radius * np.array((np.cos(time), 0.2, np.sin(time)))
        camera.transform.look_at(mesh_center)

        # TODO: take samples from random points in the orbit
        
        # Reference rendering
        rendering_pipeline.render_reference(
            camera,
            mesh,
            (texture, texture_view),
            depth_texture.create_view(),
            diffuse_texture,
            sampler,
        )
        
        # Backward pass
        rendering_pipeline.backward(network, camera.rayframe(), texture, bound, loss_buffer)
        
        loss_data = loss_buffer.to_numpy().view(np.float32)
        loss = loss_data.mean()
        history.append(loss)
        
        # Neural rendering
        if not show_reference:
            rendering_pipeline.render_neural(network, camera.rayframe(), texture, bound)
        
        # Optimize
        training_pipeline.optimize(network)
        
        # Display
        frame.blit(texture)
    
    app.run(loop)
    
    # Plot loss
    # TODO: util method
    import seaborn as sns
    from scipy.ndimage import gaussian_filter
    import matplotlib.pyplot as plt
    import os
    history_array = np.array(history)
    sns.lineplot(history_array, alpha=0.5, color="green")
    sns.lineplot(gaussian_filter(history_array, 5), linewidth=2.5, color="green")
    plt.yscale("log")
    plt.show()

if __name__ == "__main__":
    main()