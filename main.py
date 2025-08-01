import pathlib
import slangpy as spy

window = spy.Window()
device = spy.create_device(
    include_paths=[
        pathlib.Path(__file__).parent.absolute() / "slang",
    ]
)

surface = device.create_surface(window)

module = spy.Module.load_from_file(device, "main.slang")

while not window.should_close():
    window.process_events()