import slangpy as spy
from time import perf_counter
from typing import Callable
from dataclasses import dataclass
from .blitter import Blitter


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
    blitter: Blitter
    view: spy.TextureView

    def __enter__(self):
        self.window.process_events()
        self.context.new_frame(self.width, self.height)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.context.render(self.image, self.cmd)
        self.device.submit_command_buffer(self.cmd.finish())
        self.surface.present()
        self.count[0] += 1
        
    def blit(self, source: spy.Texture):
        self.blitter.blit((self.image, self.view), source)


class App(Blitter):
    def __init__(
        self,
        device: spy.Device,
        width: int = 1024,
        height: int = 1024,
        keyboard_hook: Callable[[spy.KeyboardEvent], None] = lambda _: None,
        mouse_hook: Callable[[spy.MouseEvent], None] = lambda _: None,
    ):
        super().__init__(device)
        
        self.device = device
        self.width = width
        self.height = height
        self.count = [0]
        self.keyboard_hook = keyboard_hook
        self.mouse_hook = mouse_hook

        self.window = spy.Window(width=width, height=height)
        self.surface = self.device.create_surface(self.window)
        self.surface.configure(width, height)

        self.context = spy.ui.Context(device)

        self.window.on_keyboard_event = self.keyboard_handler
        self.window.on_mouse_event = self.mouse_handler

        self.info_window = spy.ui.Window(self.context.screen, "Info", size=spy.float2(200, 100))
        self.time_text = spy.ui.Text(self.info_window)
        self.last_time = perf_counter()
        
        self.frame_views = dict()

    def alive(self) -> bool:
        return not self.window.should_close()

    def frame(self) -> Frame:
        cmd = self.device.create_command_encoder()
        image = self.surface.acquire_next_image()
        if id(image) not in self.frame_views:
            self.frame_views[id(image)] = image.create_view()
        else:
            self.frame_views[id(image)] = self.frame_views[id(image)]
        
        return Frame(
            width=self.width,
            height=self.height,
            device=self.device,
            cmd=cmd,
            count=self.count,
            context=self.context,
            image=image,
            surface=self.surface,
            window=self.window,
            blitter=self,
            view=self.frame_views[id(image)],
        )

    def run(self, loop: Callable[[Frame], None]):
        while self.alive():
            with self.frame() as frame:
                time = perf_counter() - self.last_time
                ms = time * 1000
                self.time_text.text = f"Frame time: {ms:.2f}ms"
                self.last_time = perf_counter()
                loop(frame)

    def keyboard_handler(self, event: spy.KeyboardEvent):
        self.keyboard_hook(event)
        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.escape:
                self.window.close()

    def mouse_handler(self, event: spy.MouseEvent):
        self.mouse_hook(event)
        self.context.handle_mouse_event(event)