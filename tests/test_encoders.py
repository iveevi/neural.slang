from math import e
import numpy as np
import pytest
import torch
import torch.nn as nn
from .conftest import assert_close
from common import *


def create_specialization_module(device, dim, levels, features):
    source = f"""
    import neural;
    export static const int Dim = {dim};
    export static const int Levels = {levels};
    export static const int Features = {features};
    """
    return device.load_module_from_source("specialization", source)


class DenseGrid(nn.Module):
    def __init__(self, dim, levels, features, resolutions=None):
        super().__init__()
        self.dim = dim
        self.levels = levels
        self.features = features
        
        if resolutions is None:
            resolutions = [ 2 * i + 1 for i in range(2, levels + 2)]
        self.resolutions = resolutions
        
        print("resolutions", resolutions)
        
        storages = [
                torch.randn((resolutions ** dim, features), dtype=torch.float32)
                for resolutions in self.resolutions
        ]
        
        self.storages = [
                nn.Parameter(storage)
                for storage in storages
        ]
        
        print("storages", [storage.shape for storage in self.storages])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 2
        assert input.shape[-1] == self.dim

        if self.dim == 1:        
            outputs = []
            for level, storage in enumerate(self.storages):
                scaled = input * (self.resolutions[level] - 1)
                i0 = torch.floor(scaled).long()
                i1 = torch.ceil(scaled).long()
                t = (1 - (scaled - i0)).unsqueeze(-1)
                output = t * storage[i0] + (1 - t) * storage[i1]
                outputs.append(output)
                
            return torch.cat(outputs, dim=1)
        else:
            raise NotImplementedError("Only 1D is supported for now")
        
    def to_slang(self, device: spy.Device):
        storages = [
            create_buffer_32b(device, storage.detach().cpu().numpy())
            for storage in self.storages
        ]
        print("storages", [storage.device_address for storage in storages])
        
        return {
            "resolutions": self.resolutions,
            "storages": storages,
        }


@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("dim", [1])
@pytest.mark.parametrize("levels", [2])
@pytest.mark.parametrize("features", [4])
def test_dense_grid(device, make_kernel, random_seed, dim, levels, features):
    np.random.seed(random_seed)

    specialization_module = create_specialization_module(device, dim, levels, features)
    kernel = make_kernel("dense_grid", link_modules=[specialization_module])
    
    batch_size = 16
    data = 2 * np.random.rand(batch_size, dim).astype(np.float32) - 1

    input_buffer = create_buffer_32b(device, data, dim)
    output_buffer = create_batched_buffer_32b(device, batch_size, levels * features)
    
    encoder = DenseGrid(dim, levels, features)
    
    print("input", data.shape)
    
    expected = encoder(torch.from_numpy(data))
    print("expected", expected.shape)

    kernel.dispatch(
        thread_count=(batch_size, 1, 1),
        vars={
            "globals": {
                "input": input_buffer,
                "output": output_buffer,
                **encoder.to_slang(device),
            }
        }
    )
    
    output = output_buffer.to_numpy().view(np.float32).reshape(batch_size, levels, features)
    
    print("output", output.shape)