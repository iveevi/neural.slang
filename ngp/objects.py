from __future__ import annotations
from typing import ClassVar
from dataclasses import dataclass
import slangpy as spy
import pathlib

NGP = pathlib.Path(__file__).parent

class Object:
    def dict(self) -> dict:
        raise NotImplementedError("Object must implement dict")

    def slang_type(self) -> str:
        raise NotImplementedError("Object must implement slang_type")


class Optimizer(Object):
    pass


@dataclass
class Optimizable(Object):
    PIPELINE_CACHE: ClassVar[dict[tuple[str, str], spy.ComputePipeline]] = dict()

    device: spy.Device

    @property
    def parameter_count(self):
        raise NotImplementedError("Optimizable must implement parameter_count")
    
    def alloc_optimizer_states(self, device: spy.Device, optimizer: Optimizer):
        raise NotImplementedError("Optimizable must implement alloc_optimizer_states")

    @staticmethod
    def load_update_pipeline(device: spy.Device, optimizable: Optimizable, optimizer: Optimizer, thread_count: int = 32):
        optimizable_repr = optimizable.slang_type()
        optimizer_repr = optimizer.slang_type()

        if (optimizable_repr, optimizer_repr) in Optimizable.PIPELINE_CACHE:
            return Optimizable.PIPELINE_CACHE[(optimizable_repr, optimizer_repr)]

        specialization_source = f"""
        import ngp;
        import neural;

        typealias Optimizable = {optimizable.slang_type()};
        typealias Optimizer = {optimizer.slang_type()};

        struct Globals
        {{
            Optimizable optimizable;
            Optimizer optimizer;
            RWStructuredBuffer<Optimizer.State> states;
        }}

        ParameterBlock<Globals> globals;

        [shader("compute")]
        [numthreads({thread_count}, 1, 1)]
        void update(uint3 tid : SV_DispatchThreadID)
        {{
            globals.optimizable.update(globals.optimizer, globals.states[tid.x], tid.x);
        }}
        """
        module = device.load_module_from_source("update_specialization", specialization_source)
        pipeline = device.create_compute_pipeline(device.link_program(
            modules=[module],
            entry_points=[module.entry_point("update")],
        ))
        Optimizable.PIPELINE_CACHE[(optimizable_repr, optimizer_repr)] = pipeline
        return pipeline

    def update(self, optimizer: Optimizer, optimizer_states: spy.Buffer):
        pipeline = self.load_update_pipeline(self.device, self, optimizer)

        command_encoder = self.device.create_command_encoder()
        with command_encoder.begin_compute_pass() as cmd:
            shader_object = cmd.bind_pipeline(pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.globals.optimizable = self.dict()
            cursor.globals.optimizer = optimizer.dict()
            cursor.globals.states = optimizer_states
            cmd.dispatch(thread_count=(self.parameter_count, 1, 1))

        self.device.submit_command_buffer(command_encoder.finish())


@dataclass
class MLP(Optimizable):
    input: int
    output: int
    hidden: int
    hidden_layers: int


@dataclass
class Grid(Optimizable):
    dimension: int
    features: int