import slangpy as spy
import pathlib
from typing import List, Optional


ROOT = pathlib.Path(__file__).parent.parent.absolute()


def create_device(
    additional_include_paths: Optional[List[pathlib.Path]] = None,
) -> spy.Device:
    spy.Logger.get().level = spy.LogLevel.debug
    
    device_types = [spy.DeviceType.metal, spy.DeviceType.vulkan]
    
    # Always include neural path
    include_paths = [
        ROOT / "neural",
        ROOT / "examples" / "slang",
        ROOT / "examples" / "networks" / "slang", # TODO: remove
        ROOT / "ngp" / "slang",
    ]
    
    if additional_include_paths:
        include_paths.extend(additional_include_paths)
    
    last_error = None
    for device_type in device_types:
        try:
            return spy.create_device(
                device_type,
                enable_debug_layers=True,
                include_paths=include_paths,
            )
        except Exception as e:
            last_error = e
            continue
    
    raise RuntimeError(f"Failed to create device with any of {device_types}. Last error: {last_error}")
    
    
def create_compute_pipeline(
    device: spy.Device,
    module: spy.SlangModule,
    specialization_modules: List[spy.SlangModule],
    entry_point: str,
) -> spy.ComputePipeline:
    program = device.link_program(
        modules=[module] + specialization_modules,
        entry_points=[module.entry_point(entry_point)],
    )
    return device.create_compute_pipeline(program)