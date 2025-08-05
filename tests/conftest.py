
import pathlib
import pytest
import slangpy as spy
import numpy as np


@pytest.fixture(scope="session")
def device():
    return spy.create_device(
        include_paths=[
            pathlib.Path(__file__).parent.parent.absolute() / "slang",
        ],
    )


@pytest.fixture
def make_kernel(device):
    def _make_kernel(entry_point_name):
        program = device.load_program(
            "tests/main.slang",
            entry_point_names=[entry_point_name],
        )
        return device.create_compute_kernel(program)
    return _make_kernel


@pytest.fixture(autouse=True)
def setup_numpy():
    np.set_printoptions(threshold=10000, linewidth=10000)


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


# Export the helper function so tests can import it
__all__ = ['assert_close']