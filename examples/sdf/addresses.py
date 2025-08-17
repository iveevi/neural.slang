from __future__ import annotations
import slangpy as spy
from common import *
from ..networks.addresses import Network
from ..util import *
from ..ngp.objects import Object, Optimizer, MLP


HERE = ROOT / "examples" / "sdf"


# TODO: move to main as a mlp agnostic pipeline
# TODO: decorators for extracting each part of the pipeline -- reflection to get the global variables
class RenderingPipeline:
    @staticmethod
    def load_specialization_module(device: spy.Device, network: Network):
        source = f"""
        import neural;
        import mlp;
        
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

        self.render_heatmap_pipeline = create_compute_pipeline(device, self.module, [self.specialization_module], "render_heatmap")
        self.render_normal_pipeline = create_compute_pipeline(device, self.module, [self.specialization_module], "render_normal")

        self.backward_pipeline = create_compute_pipeline(device, self.module, [self.specialization_module], "backward")
        self.update_mlp_pipeline = create_compute_pipeline(device, self.module, [self.specialization_module], "update_mlp")
        self.update_grid_pipeline = create_compute_pipeline(device, self.module, [self.specialization_module], "update_grid")
        
    def render_heatmap(self, mlp: MLP, grid: Object, rayframe: RayFrame, target_texture: spy.Texture):
        command_encoder = self.device.create_command_encoder()

        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.render_heatmap_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.mlp = mlp.dict()
            cursor.grid = grid.dict()
            cursor.rayFrame = rayframe.dict()
            cursor.targetTexture = target_texture
            cursor.targetResolution = (target_texture.width, target_texture.height)
            cmd.dispatch(thread_count=(target_texture.width, target_texture.height, 1))

        self.device.submit_command_buffer(command_encoder.finish())
        
    def render_normal(self, mlp: MLP, grid: Object, rayframe: RayFrame, target_texture: spy.Texture):
        command_encoder = self.device.create_command_encoder()

        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.render_normal_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.mlp = mlp.dict()
            cursor.grid = grid.dict()
            cursor.rayFrame = rayframe.dict()
            cursor.targetTexture = target_texture
            cursor.targetResolution = (target_texture.width, target_texture.height)
            cmd.dispatch(thread_count=(target_texture.width, target_texture.height, 1))

        self.device.submit_command_buffer(command_encoder.finish())

    def backward(self, mlp: MLP, grid: Object, input_buffer: spy.Buffer, expected_buffer: spy.Buffer, loss_buffer: spy.Buffer, sample_count: int):
        command_encoder = self.device.create_command_encoder()

        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.backward_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.mlp = mlp.dict()
            cursor.grid = grid.dict()
            cursor.inputBuffer = input_buffer
            cursor.expectedBuffer = expected_buffer
            cursor.lossBuffer = loss_buffer
            cursor.boost = 1.0 / sample_count
            cmd.dispatch(thread_count=(sample_count, 1, 1))

        self.device.submit_command_buffer(command_encoder.finish())

    def update_mlp(self, mlp: MLP, optimizer: Optimizer, optimizer_states: spy.Buffer, parameter_count: int):
        command_encoder = self.device.create_command_encoder()

        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.update_mlp_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.mlp = mlp.dict()
            cursor.optimizer = optimizer.dict()
            cursor.optimizerStates = optimizer_states
            cmd.dispatch(thread_count=(parameter_count, 1, 1))

        self.device.submit_command_buffer(command_encoder.finish())

    def update_grid(self, grid: Object, optimizer: Optimizer, optimizer_states: spy.Buffer, parameter_count: int):
        command_encoder = self.device.create_command_encoder()

        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.update_grid_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.grid = grid.dict()
            cursor.optimizer = optimizer.dict()
            cursor.optimizerStates = optimizer_states
            cmd.dispatch(thread_count=(parameter_count, 1, 1))

        self.device.submit_command_buffer(command_encoder.finish())
