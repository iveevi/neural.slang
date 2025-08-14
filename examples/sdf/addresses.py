from __future__ import annotations
import slangpy as spy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from time import perf_counter
import pytorch_volumetric as pv
from scipy.ndimage import gaussian_filter
from common import *
from ..networks.addresses import Network, TrainingPipeline
from ..render_util import *

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
        SOURCE = ROOT / "examples" / "slang" / "neural_sdf.slang"
        
        self.device = device
        self.module = device.load_module(str(SOURCE))
        self.specialization_module = self.load_specialization_module(device, network)

        self.render_neural_pipeline = create_compute_pipeline(device, self.module, [self.specialization_module], "render_neural")

    def render_neural(self, network: Network, rayframe: RayFrame, target_texture: spy.Texture):
        command_encoder = self.device.create_command_encoder()

        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(self.render_neural_pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.network = network.dict()
            cursor.rayFrame = rayframe.dict()
            cursor.targetTexture = target_texture
            cursor.targetResolution = (target_texture.width, target_texture.height)
            cmd.dispatch(thread_count=(target_texture.width, target_texture.height, 1))

        self.device.submit_command_buffer(command_encoder.finish())