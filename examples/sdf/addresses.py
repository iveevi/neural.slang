from __future__ import annotations
import slangpy as spy
from common import *
from ..networks.addresses import Network, TrainingPipeline
from ..util import *


HERE = ROOT / "examples" / "sdf"


class RenderingPipeline:
    @staticmethod
    def load_specialization_module(device: spy.Device, network: Network):
        source = f"""
        import neural;
        import mlp;
        
        export static const int Hidden = {network.hidden};
        export static const int HiddenLayers = {network.hidden_layers};
        export static const int Levels = {network.levels};
        export struct MLP : IMLP<float, 3, 1> = AddressBasedMLP<3, 1, 32, 2, ReLU<float>, ReLU<float>>;
        """
        return device.load_module_from_source("specialization", source)

    def __init__(self, device: spy.Device, network: Network):
        SOURCE = HERE / "slang" / "main.slang"
        
        self.device = device
        self.module = device.load_module(str(SOURCE))
        self.specialization_module = self.load_specialization_module(device, network)

        self.render_heatmap_pipeline = create_compute_pipeline(device, self.module, [self.specialization_module], "render_heatmap")
        self.render_normal_pipeline = create_compute_pipeline(device, self.module, [self.specialization_module], "render_normal")
        
        layout = self.module.layout
        mlp_type = layout.find_type_by_name("AddressBasedMLP<3, 1, 32, 2, ReLU<float>, ReLU<float>>")
        print("mlp_type:", mlp_type)
        self.mlp_layout = layout.get_type_layout(mlp_type)
        print("mlp_layout:", self.mlp_layout)
        
    def render_heatmap(self, network: Network, rayframe: RayFrame, target_texture: spy.Texture):
        command_encoder = self.device.create_command_encoder()

        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.render_heatmap_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.network = network.dict()
            cursor.rayFrame = rayframe.dict()
            cursor.targetTexture = target_texture
            cursor.targetResolution = (target_texture.width, target_texture.height)
            cmd.dispatch(thread_count=(target_texture.width, target_texture.height, 1))

        self.device.submit_command_buffer(command_encoder.finish())
        
    def render_normal(self, network: Network, rayframe: RayFrame, target_texture: spy.Texture):
        command_encoder = self.device.create_command_encoder()

        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.render_normal_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.network = network.dict()
            cursor.rayFrame = rayframe.dict()
            cursor.targetTexture = target_texture
            cursor.targetResolution = (target_texture.width, target_texture.height)
            cmd.dispatch(thread_count=(target_texture.width, target_texture.height, 1))

        self.device.submit_command_buffer(command_encoder.finish())