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
    
    def view_matrix(self) -> np.ndarray:
        right, up, forward = self.transform.axes()
        position = self.transform.position
        
        view = np.eye(4, dtype=np.float32)
        
        view[0, 0:3] = right
        view[1, 0:3] = up
        view[2, 0:3] = -forward
        
        view[0, 3] = -np.dot(right, position)
        view[1, 3] = -np.dot(up, position)
        view[2, 3] = np.dot(forward, position)
        
        return view
    
    def perspective_matrix(self) -> np.ndarray:
        fov_rad = np.radians(self.fov)
        f = 1.0 / np.tan(fov_rad / 2.0)
        
        proj = np.zeros((4, 4), dtype=np.float32)
        
        proj[0, 0] = f / self.aspect_ratio
        proj[1, 1] = f
        proj[2, 2] = (self.far + self.near) / (self.near - self.far)
        proj[2, 3] = (2.0 * self.far * self.near) / (self.near - self.far)
        proj[3, 2] = -1.0
        
        return proj