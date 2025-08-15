import slangpy as spy
from pathlib import Path
from typing import Any
from common import *

class Blitter:
    def __init__(self, device: spy.Device):
        self.device = device
        self.module = device.load_module(str(ROOT / "examples" / "slang" / "blit.slang"))
        
        upscaled_program = device.link_program(
            modules=[self.module],
            entry_points=[
                self.module.entry_point("upscaled_vertex"),
                self.module.entry_point("upscaled_fragment"),
            ],
        )
        
        input_layout = device.create_input_layout(
            input_elements=[
                {
                    "semantic_name": "POSITION",
                    "semantic_index": 0,
                    "format": spy.Format.rg32_float,
                    "offset": 0,
                    "buffer_slot_index": 0,
                }
            ],
            vertex_streams=[
                { "stride": 4 * 2 }
            ]
        )
        
        self.pipeline = device.create_render_pipeline(
            program=upscaled_program,
            input_layout=input_layout,
            primitive_topology=spy.PrimitiveTopology.triangle_list,
            targets=[
                { "format": spy.Format.bgra8_unorm_srgb },
            ],
        )
        
        quad_vertices = np.array([
            [-1.0, -1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
        ], dtype=np.float32)
        
        quad_triangles = np.array([
            [0, 1, 2],
            [1, 3, 2],
        ], dtype=np.uint32)
        
        self.quad_vertices_buffer = device.create_buffer(
            data=quad_vertices,
            usage=spy.BufferUsage.vertex_buffer,
        )
        
        self.quad_triangles_buffer = device.create_buffer(
            data=quad_triangles,
            usage=spy.BufferUsage.index_buffer,
        )
        
        self.sampler = device.create_sampler(
            min_filter=spy.TextureFilteringMode.linear,
            mag_filter=spy.TextureFilteringMode.linear,
            mip_filter=spy.TextureFilteringMode.linear,
            address_u=spy.TextureAddressingMode.wrap,
            address_v=spy.TextureAddressingMode.wrap,
            address_w=spy.TextureAddressingMode.wrap,
        )
        
    def blit(self, destination: tuple[spy.Texture, spy.TextureView], source: spy.Texture):
        command_encoder = self.device.create_command_encoder()
        
        render_pass_args: Any = {
            "color_attachments": [
                {
                    "view": destination[1],
                    "clear_value": [0.0, 0.0, 0.0, 1.0],
                    "load_op": spy.LoadOp.clear,
                    "store_op": spy.StoreOp.store,
                }
            ],
        }
        
        with command_encoder.begin_render_pass(render_pass_args) as cmd:
            shader_object = cmd.bind_pipeline(self.pipeline)
            
            cursor = spy.ShaderCursor(shader_object)
            cursor.referenceTexture = source
            cursor.referenceSampler = self.sampler
            cursor.upscaledResolution = (destination[0].width, destination[0].height)
            
            state = spy.RenderState()
            state.vertex_buffers = [ self.quad_vertices_buffer ]
            state.index_buffer = self.quad_triangles_buffer
            state.index_format = spy.IndexFormat.uint32
            state.viewports = [
                spy.Viewport.from_size(destination[0].width, destination[0].height)
            ]
            state.scissor_rects = [
                spy.ScissorRect.from_size(destination[0].width, destination[0].height)
            ]
            
            cmd.set_render_state(state)
            
            params = spy.DrawArguments()
            params.instance_count = 1
            params.vertex_count = 6

            cmd.draw_indexed(params)
        
        self.device.submit_command_buffer(command_encoder.finish())