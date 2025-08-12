import pathlib
import pytest
import slangpy as spy
import numpy as np
from common.util import *


@pytest.fixture(scope="function")
def device():
    device = create_device()
    try:
        yield device
    finally:
        # Best-effort explicit teardown to avoid native leaks
        for method_name in [
            "wait_idle",
            "finish",
            "flush",
            "collect_garbage",
            "gc",
            "close",
            "shutdown",
            "destroy",
        ]:
            method = getattr(device, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass

        # Drop Python references and force a GC cycle
        device = None
        import gc
        gc.collect()


@pytest.fixture
def make_kernel(device):
    def _make_kernel(shader_name, link_modules=None):
        if link_modules is None:
            link_modules = []
        if not shader_name.endswith('.slang'):
            shader_file = f"tests/slang/{shader_name}.slang"
        else:
            shader_file = f"tests/slang/{shader_name}"
        
        main_module = device.load_module(shader_file)
        entry_point = main_module.entry_point("computeMain")
        program = device.link_program(
            modules=[main_module] + link_modules,
            entry_points=[entry_point],
        )
        return device.create_compute_kernel(program)
    return _make_kernel


@pytest.fixture(autouse=True)
def setup_numpy():
    np.set_printoptions(threshold=10000, linewidth=10000)


# Force a GC pass after each test to reduce peak memory from native handles
@pytest.fixture(autouse=True)
def force_gc():
    yield
    import gc
    gc.collect()


def assert_close(actual, expected, rtol=1e-5, atol=1e-6):
    error = np.abs(actual - expected).sum()
    max_diff = np.max(np.abs(actual - expected))
    
    # Use both absolute and relative tolerance checks
    is_close = np.allclose(actual, expected, rtol=rtol, atol=atol)
    
    if not is_close:
        print(f"Arrays not close enough:")
        print(f"Max difference: {max_diff}")
        print(f"Total error: {error}")
        print(f"Actual: {actual}")
        print(f"Expected: {expected}")
        
    assert is_close, f"Arrays differ by more than tolerance (max_diff={max_diff}, total_error={error})"


__all__ = ['assert_close']