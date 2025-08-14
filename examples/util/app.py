import slangpy as spy
from typing import Callable
from dataclasses import dataclass


@dataclass
class Frame:
    width: int
    height: int
    device: spy.Device
    cmd: spy.CommandEncoder
    count: list[int]
    context: spy.ui.Context
    image: spy.Texture
    surface: spy.Surface
    window: spy.Window

    def __enter__(self):
        self.window.process_events()
        self.context.new_frame(self.width, self.height)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.context.render(self.image, self.cmd)
        self.device.submit_command_buffer(self.cmd.finish())
        self.surface.present()
        self.count[0] += 1


class App:
    def __init__(self, device: spy.Device, width: int = 512, height: int = 512):
        self.device = device
        self.width = width
        self.height = height
        self.count = [0]

        self.window = spy.Window(width=width, height=height)
        self.surface = self.device.create_surface(self.window)
        self.surface.configure(width, height)

        self.context = spy.ui.Context(device)

        self.window.on_keyboard_event = self.keyboard_handler
        self.window.on_mouse_event = self.mouse_handler

    def alive(self) -> bool:
        return not self.window.should_close()

    def frame(self) -> Frame:
        cmd = self.device.create_command_encoder()
        image = self.surface.acquire_next_image()
        return Frame(
            width=self.width,
            height=self.height,
            device=self.device,
            cmd=cmd,
            count=self.count,
            context=self.context,
            image=image,
            surface=self.surface,
            window=self.window
        )

    def run(self, loop: Callable[[Frame], None]):
        while self.alive():
            with self.frame() as frame:
                loop(frame)

    def keyboard_handler(self, event: spy.KeyboardEvent):
        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.escape:
                self.window.close()

    def mouse_handler(self, event: spy.MouseEvent):
        self.context.handle_mouse_event(event)