from ..networks.addresses import Network, TrainingPipeline
from ..util import *
from common import *
import slangpy as spy

HERE = ROOT / "examples" / "volume"


class RenderingPipeline:
    def __init__(self, device: spy.Device, network: Network):
        SOURCE = HERE / "slang" / "main.slang"
        
        self.device = device
        self.module = device.load_module(str(SOURCE))
        
        # Reference pipeline
        reference_program = device.load_program(
            module_name=str(SOURCE),
            entry_point_names=["render_reference"],
        )
        
        reference_pipeline = device.create_ray_tracing_pipeline(
            program=reference_program,
            hit_groups=[],
            max_recursion=1,
            max_ray_payload_size=16,
        )
        
        reference_table = device.create_shader_table(
            program=reference_program,
            ray_gen_entry_points=["render_reference"],
        )
        
        self.reference_pipeline = (reference_pipeline, reference_table)

    def render_reference(self, rayframe: RayFrame, target_texture: spy.Texture):
        command_encoder = self.device.create_command_encoder()
        
        with command_encoder.begin_ray_tracing_pass() as cmd:
            shader_object = cmd.bind_pipeline(*self.reference_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.rayFrame = rayframe.dict()
            cursor.targetTexture = target_texture
            cmd.dispatch_rays(0, dimensions=(target_texture.width, target_texture.height, 1))
            
        self.device.submit_command_buffer(command_encoder.finish())
        