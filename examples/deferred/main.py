from ..util import *
from PIL import Image
from common import *
from dataclasses import dataclass
from ngp import AddressBasedMLP, MLP, Adam, Optimizer, DenseGrid
from typing import Any
import numpy as np
import slangpy as spy
import trimesh
import time


HERE = ROOT / "examples" / "deferred"


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

        # Reference rendering pipeline
        self.reference_pipeline = self.create_rasterization_pipeline(
            device,
            "reference_fragment",
            [
                { "format": spy.Format.rgba32_float },
            ]
        )

        # G-buffer pipeline
        self.gbuffer_pipeline = self.create_rasterization_pipeline(
            device,
            "gbuffer_fragment",
            [
                # Position
                { "format": spy.Format.rgba32_float },
                # Albedo
                { "format": spy.Format.rgba32_float },
                # Normal
                { "format": spy.Format.rgba32_float },
            ]
        )

        # Neural shading pipeline
        self.neural_shading_pipeline = create_compute_pipeline(device, self.module, [self.specialization_module], "deferred_neural_shading")
        self.neural_illumination_pipeline = create_compute_pipeline(device, self.module, [self.specialization_module], "deferred_neural_illumination")

        # Backward pass pipeline
        self.backward_pipeline = create_compute_pipeline(device, self.module, [self.specialization_module], "backward")

        # Update pipeline
        self.update_pipeline = create_compute_pipeline(device, self.module, [self.specialization_module], "update_stable")

    def create_rasterization_pipeline(self, device: spy.Device, fragment_shader: str, targets: list[dict[str, spy.Format]]):
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
            targets=targets,
            depth_stencil={
                "format": spy.Format.d32_float,
                "depth_test_enable": True,
                "depth_write_enable": True,
                "depth_func": spy.ComparisonFunc.less,
            }
        )

    @staticmethod
    def create_render_pass_args(targets: list[tuple[spy.Texture, spy.TextureView]], depth_target: spy.TextureView):
        return {
            "color_attachments": [
                {
                    "view": target[1],
                        "clear_value": [0.0, 0.0, 0.0, 1.0],
                        "load_op": spy.LoadOp.clear,
                        "store_op": spy.StoreOp.store,
                }
                for target in targets
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
                         blas: spy.AccelerationStructure,
                         target: tuple[spy.Texture, spy.TextureView],
                         depth_target: spy.TextureView,
                         diffuse_texture: spy.Texture,
                         diffuse_sampler: spy.Sampler,
                         seed_texture: spy.Texture,
                         seed_sampler: spy.Sampler):
        render_pass_args: Any = self.create_render_pass_args([target], depth_target)
        
        command_encoder = self.device.create_command_encoder()
        
        with command_encoder.begin_render_pass(render_pass_args) as cmd:
            shader_object = cmd.bind_pipeline(self.reference_pipeline)
            
            cursor = spy.ShaderCursor(shader_object)
            cursor.view = camera.view_matrix()
            cursor.perspective = camera.perspective_matrix()
            cursor.diffuseTexture = diffuse_texture
            cursor.diffuseSampler = diffuse_sampler
            cursor.blas = blas
            cursor.seedTexture = seed_texture
            cursor.seedSampler = seed_sampler
            cursor.targetResolution = spy.float2(target[0].width, target[0].height)
            cursor.time = time.perf_counter()
            
            state = self.create_render_state(mesh, target)
            cmd.set_render_state(state)
            
            params = spy.DrawArguments()
            params.instance_count = 1
            params.vertex_count = mesh.triangle_count * 3
            cmd.draw_indexed(params)
            
        self.device.submit_command_buffer(command_encoder.finish())

    def render_gbuffer(self,
                        camera: Camera,
                        mesh: TriangleMesh,
                        depth_target: spy.TextureView,
                        diffuse_texture: spy.Texture,
                        diffuse_sampler: spy.Sampler,
                        gbuffer_position: tuple[spy.Texture, spy.TextureView],
                        gbuffer_albedo: tuple[spy.Texture, spy.TextureView],
                        gbuffer_normal: tuple[spy.Texture, spy.TextureView]):
        render_pass_args: Any = self.create_render_pass_args(
            [gbuffer_position, gbuffer_albedo, gbuffer_normal],
            depth_target
        )

        command_encoder = self.device.create_command_encoder()

        with command_encoder.begin_render_pass(render_pass_args) as cmd:
            shader_object = cmd.bind_pipeline(self.gbuffer_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.view = camera.view_matrix()
            cursor.perspective = camera.perspective_matrix()
            cursor.diffuseTexture = diffuse_texture
            cursor.diffuseSampler = diffuse_sampler
            state = self.create_render_state(mesh, gbuffer_position)
            cmd.set_render_state(state)
            params = spy.DrawArguments()
            params.instance_count = 1
            params.vertex_count = mesh.triangle_count * 3
            cmd.draw_indexed(params)

        self.device.submit_command_buffer(command_encoder.finish())

    def render_neural_shading(self,
                              mlp: MLP,
                              grid: DenseGrid,
                              target: tuple[spy.Texture, spy.TextureView],
                              gbuffer_position: tuple[spy.Texture, spy.TextureView],
                              gbuffer_albedo: tuple[spy.Texture, spy.TextureView],
                              gbuffer_normal: tuple[spy.Texture, spy.TextureView]):
        command_encoder = self.device.create_command_encoder()
        
        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.neural_shading_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.mlp = mlp.dict()
            cursor.grid = grid.dict()
            cursor.gbufferPositionTexture = gbuffer_position[0]
            cursor.gbufferAlbedoTexture = gbuffer_albedo[0]
            cursor.gbufferNormalTexture = gbuffer_normal[0]
            cursor.targetTexture = target[0]
            cmd.dispatch(thread_count=(target[0].width, target[0].height, 1))
            
        self.device.submit_command_buffer(command_encoder.finish())

    def render_neural_illumination(self,
                              mlp: MLP,
                              grid: DenseGrid,
                              target: tuple[spy.Texture, spy.TextureView],
                              gbuffer_position: tuple[spy.Texture, spy.TextureView],
                              gbuffer_albedo: tuple[spy.Texture, spy.TextureView],
                              gbuffer_normal: tuple[spy.Texture, spy.TextureView]):
        command_encoder = self.device.create_command_encoder()

        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.neural_illumination_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.mlp = mlp.dict()
            cursor.grid = grid.dict()
            cursor.gbufferPositionTexture = gbuffer_position[0]
            cursor.gbufferAlbedoTexture = gbuffer_albedo[0]
            cursor.gbufferNormalTexture = gbuffer_normal[0]
            cursor.targetTexture = target[0]
            cmd.dispatch(thread_count=(target[0].width, target[0].height, 1))

        self.device.submit_command_buffer(command_encoder.finish())

    def backward(self,
                 mlp: MLP,
                 grid: DenseGrid,
                 reference_texture: spy.Texture,
                 gbuffer_position: tuple[spy.Texture, spy.TextureView],
                 gbuffer_albedo: tuple[spy.Texture, spy.TextureView],
                 gbuffer_normal: tuple[spy.Texture, spy.TextureView],
                 boost: float):
        command_encoder = self.device.create_command_encoder()
        
        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.backward_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.mlp = mlp.dict()
            cursor.grid = grid.dict()
            cursor.referenceTexture = reference_texture
            cursor.gbufferPositionTexture = gbuffer_position[0]
            cursor.gbufferAlbedoTexture = gbuffer_albedo[0]
            cursor.gbufferNormalTexture = gbuffer_normal[0]
            cursor.boost = boost
            cmd.dispatch(thread_count=(gbuffer_position[0].width, gbuffer_position[0].height, 1))
            
        self.device.submit_command_buffer(command_encoder.finish())

    def update_stable(self,
                      stable_parameters: spy.Buffer,
                      trained_parameters: spy.Buffer,
                      parameter_count: int,
                      eta: float):
        command_encoder = self.device.create_command_encoder()

        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.update_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.stableParameters = stable_parameters
            cursor.trainedParameters = trained_parameters
            cursor.eta = eta
            cmd.dispatch(thread_count=(parameter_count, 1, 1))
            
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


def build_tlas(device: spy.Device, blas: spy.AccelerationStructure):
    instance_list = device.create_acceleration_structure_instance_list(1)

    xform = spy.math.matrix_from_translation(spy.float3(0.0, 0.0, 0.0))
    xform = spy.float3x4(xform)

    instance_desc = spy.AccelerationStructureInstanceDesc()
    instance_desc.transform = xform
    instance_desc.instance_id = 0
    instance_desc.instance_mask = 0xff
    instance_desc.instance_contribution_to_hit_group_index = 0
    instance_desc.flags = spy.AccelerationStructureInstanceFlags.none
    instance_desc.acceleration_structure = blas.handle

    instance_list.write(0, instance_desc)

    tlas_build_desc = spy.AccelerationStructureBuildDesc(
        {
            "inputs": [ instance_list.build_input_instances() ]
        }
    )

    tlas_sizes = device.get_acceleration_structure_sizes(tlas_build_desc)

    tlas_scratch_buffer = device.create_buffer(
        size=tlas_sizes.scratch_size,
        usage=spy.BufferUsage.unordered_access,
    )
    
    tlas_buffer = device.create_buffer(
        size=tlas_sizes.acceleration_structure_size,
        usage=spy.BufferUsage.acceleration_structure,
    )

    tlas = device.create_acceleration_structure(
        size=tlas_buffer.size,
    )

    command_encoder = device.create_command_encoder()
    command_encoder.build_acceleration_structure(
        desc=tlas_build_desc,
        dst=tlas,
        src=None,
        scratch_buffer=tlas_scratch_buffer,
    )
    device.submit_command_buffer(command_encoder.finish())

    return tlas


def generate_blue_noise_texture(size: int) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    from scipy.ndimage import distance_transform_edt
    
    white_noise = np.random.rand(size, size).astype(np.float32)
    
    sigma = size / 32
    low_freq = gaussian_filter(white_noise, sigma, mode='wrap')
    blue_noise = white_noise - low_freq
    
    blue_noise = (blue_noise - blue_noise.min()) / (blue_noise.max() - blue_noise.min())
    
    threshold = 0.5
    dithered = (blue_noise > threshold).astype(np.float32)
    
    dist_field = distance_transform_edt(1 - dithered)
    dist_field = np.minimum(dist_field, distance_transform_edt(dithered))
    
    if dist_field.max() > 0:
        dist_field = dist_field / dist_field.max()
    
    result = (dist_field * 0.7 + white_noise * 0.3) % 1.0
    result = gaussian_filter(result, 0.5, mode='wrap')
    result = (result - result.min()) / (result.max() - result.min())
    return result.astype(np.float32)


def main():
    device = create_device()

    optimizer = Adam(alpha=1e-3)
    
    mlp_stable = AddressBasedMLP.new(device, hidden=64, hidden_layers=3, input=11, output=3)
    # TODO: multires grid...
    grid_stable = DenseGrid.new(device, dimension=3, features=8, resolution=64)

    mlp_train = AddressBasedMLP.new(device, hidden=64, hidden_layers=3, input=11, output=3)
    # TODO: multires grid...
    grid_train = DenseGrid.new(device, dimension=3, features=8, resolution=64)

    mlp_optimizer_states = mlp_train.alloc_optimizer_states(device, optimizer)
    grid_optimizer_states = grid_train.alloc_optimizer_states(device, optimizer)
    
    rendering_pipeline = RenderingPipeline(device, mlp_train)
    
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
    blas = mesh.blas(device)
    tlas = build_tlas(device, blas)
    
    pause_orbit = False
    show_mode = "reference"
    vertical = 0.0
    
    def keyboard_hook(event: spy.KeyboardEvent):
        nonlocal show_mode
        nonlocal pause_orbit
        nonlocal vertical
        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.tab:
                match show_mode:
                    case "reference":
                        show_mode = "neural_shading"
                    case "neural_shading":
                        show_mode = "neural_illumination"
                    case "neural_illumination":
                        show_mode = "reference"
                    case _:
                        show_mode = "reference"
            if event.key == spy.KeyCode.key1:
                show_mode = "reference"
            if event.key == spy.KeyCode.key2:
                show_mode = "neural_shading"
            if event.key == spy.KeyCode.key3:
                show_mode = "neural_illumination"
            if event.key == spy.KeyCode.space:
                pause_orbit = not pause_orbit
        elif event.type == spy.KeyboardEventType.key_repeat:
            if event.key == spy.KeyCode.up:
                vertical += 0.01
            if event.key == spy.KeyCode.down:
                vertical -= 0.01

    app = App(device, keyboard_hook=keyboard_hook)

    # Generate blue noise texture for better sampling patterns
    seed_resolution = 1024

    seed_texture = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.r32_float,
        width=seed_resolution,
        height=seed_resolution,
        usage=spy.TextureUsage.shader_resource,
        data=generate_blue_noise_texture(seed_resolution),
    )

    seed_sampler = device.create_sampler(
        min_filter=spy.TextureFilteringMode.linear,
        mag_filter=spy.TextureFilteringMode.linear,
        mip_filter=spy.TextureFilteringMode.linear,
        address_u=spy.TextureAddressingMode.wrap,
        address_v=spy.TextureAddressingMode.wrap,
        address_w=spy.TextureAddressingMode.wrap,
    )

    # Final render resources    
    normal_resolution = 1024
    texture, texture_view = alloc_target_texture(device, normal_resolution, normal_resolution)
    gb_position_texture, gb_position_texture_view = alloc_target_texture(device, normal_resolution, normal_resolution)
    gb_albedo_texture, gb_albedo_texture_view = alloc_target_texture(device, normal_resolution, normal_resolution)
    gb_normal_texture, gb_normal_texture_view = alloc_target_texture(device, normal_resolution, normal_resolution)

    # Training resources
    train_resolution = 256
    train_ref_texture, train_ref_texture_view = alloc_target_texture(device, train_resolution, train_resolution)
    train_gb_position_texture, train_gb_position_texture_view = alloc_target_texture(device, train_resolution, train_resolution)
    train_gb_albedo_texture, train_gb_albedo_texture_view = alloc_target_texture(device, train_resolution, train_resolution)
    train_gb_normal_texture, train_gb_normal_texture_view = alloc_target_texture(device, train_resolution, train_resolution)

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
    
    counter = 0

    def set_orbit_camera(time: float):
        orbit_radius = mesh_size * 1.2  # Orbit radius based on mesh size
        camera.transform.position = mesh_center + orbit_radius * np.array(
            (np.cos(time),
            vertical,
            np.sin(time))
        )
        camera.transform.look_at(mesh_center)

    # TODO: fix camera, rotate object?
    sample_count = 0
    def loop(frame: Frame):
        nonlocal sample_count
        sample_count += 1

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
            tlas,
            (train_ref_texture, train_ref_texture_view),
            depth_texture_view,
            diffuse_texture,
            sampler,
            seed_texture,
            seed_sampler,
        )

        # Prepare inputs for neural shading
        rendering_pipeline.render_gbuffer(
            camera,
            mesh,
            depth_texture_view,
            diffuse_texture,
            sampler,
            (train_gb_position_texture, train_gb_position_texture_view),
            (train_gb_albedo_texture, train_gb_albedo_texture_view),
            (train_gb_normal_texture, train_gb_normal_texture_view),
        )

        # Backward pass
        rendering_pipeline.backward(
            mlp_train,
            grid_train,
            train_ref_texture,
            (train_gb_position_texture, train_gb_position_texture_view),
            (train_gb_albedo_texture, train_gb_albedo_texture_view),
            (train_gb_normal_texture, train_gb_normal_texture_view),
            boost=(1.0 / (train_ref_texture.width * train_ref_texture.height))
        )
        
        # Rendering
        if show_mode == "reference":
            rendering_pipeline.render_reference(
                camera,
                mesh,
                tlas,
                (texture, texture_view),
                depth_texture_view,
                diffuse_texture,
                sampler,
                seed_texture,
                seed_sampler,
            )
        else:
            rendering_pipeline.render_gbuffer(
                camera,
                mesh,
                depth_texture_view,
                diffuse_texture,
                sampler,
                (gb_position_texture, gb_position_texture_view),
                (gb_albedo_texture, gb_albedo_texture_view),
                (gb_normal_texture, gb_normal_texture_view),
            )

            if show_mode == "neural_shading":
                rendering_pipeline.render_neural_shading(
                    mlp_stable,
                    grid_stable,
                    (texture, texture_view),
                    (gb_position_texture, gb_position_texture_view),
                    (gb_albedo_texture, gb_albedo_texture_view),
                    (gb_normal_texture, gb_normal_texture_view),
                )
            elif show_mode == "neural_illumination":
                rendering_pipeline.render_neural_illumination(
                    mlp_stable,
                    grid_stable,
                    (texture, texture_view),
                    (gb_position_texture, gb_position_texture_view),
                    (gb_albedo_texture, gb_albedo_texture_view),
                    (gb_normal_texture, gb_normal_texture_view),
                )

        # Optimize
        mlp_train.update(optimizer, mlp_optimizer_states)
        grid_train.update(optimizer, grid_optimizer_states)

        # Update stable parameters
        eta = 0.01
        rendering_pipeline.update_stable(mlp_stable.parameter_buffer, mlp_train.parameter_buffer, mlp_stable.parameter_count, eta=eta)
        rendering_pipeline.update_stable(grid_stable.parameter_buffer, grid_train.parameter_buffer, grid_stable.parameter_count, eta=eta)

        # Display
        frame.blit(texture)
    
    app.run(loop)


if __name__ == "__main__":
    main()