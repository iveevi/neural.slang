import numpy as np
import slangpy as spy
from dataclasses import dataclass
from .transform import Transform

@dataclass
class RayFrame:
    origin: np.ndarray
    lower_left: np.ndarray
    horizontal: np.ndarray
    vertical: np.ndarray
    
    def dict(self):
        return {
            "origin": spy.float3(self.origin.tolist()),
            "lower_left": spy.float3(self.lower_left.tolist()),
            "horizontal": spy.float3(self.horizontal.tolist()),
            "vertical": spy.float3(self.vertical.tolist()),
        }


@dataclass
class Camera:
    transform: Transform
    fov: float
    aspect_ratio: float
    near: float
    far: float
    
    @staticmethod
    def new(transform: Transform = Transform.new(),
            fov: float = 45,
            aspect_ratio: float = 1,
            near: float = 0.1,
            far: float = 100):
        return Camera(transform, fov, aspect_ratio, near, far)
    
    def rayframe(self) -> RayFrame:
        normalize = lambda x: x / np.linalg.norm(x)
        
        _, up, forward = self.transform.axes()
        
        vfov = np.radians(self.fov)
        h = np.tan(vfov / 2)
        vheight = 2 * h
        vwidth = vheight * self.aspect_ratio
        
        w = normalize(-forward)
        u = normalize(np.cross(up, w))
        v = np.cross(w, u)

        horizontal = u * vwidth
        vertical = v * vheight

        return RayFrame(
            origin=self.transform.position,
            lower_left=self.transform.position - horizontal/2 - vertical/2 - w,
            horizontal=horizontal,
            vertical=vertical,
        )