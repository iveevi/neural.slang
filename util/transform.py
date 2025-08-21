import numpy as np
from dataclasses import dataclass


@dataclass
class Transform:
    position: np.ndarray
    rotation: np.ndarray
    scale: np.ndarray

    @staticmethod
    def new(position: np.ndarray = np.array((0.0, 0.0, 0.0)),
            rotation: np.ndarray = np.array((0.0, 0.0, 0.0)),
            scale: np.ndarray = np.array((1.0, 1.0, 1.0))):
        return Transform(position, rotation, scale)
    
    def _rotation_matrix(self) -> np.ndarray:
        rx, ry, rz = np.radians(self.rotation)
        
        cos_x, sin_x = np.cos(rx), np.sin(rx)
        cos_y, sin_y = np.cos(ry), np.sin(ry)
        cos_z, sin_z = np.cos(rz), np.sin(rz)
        
        Rx = np.array([
            [1, 0, 0],
            [0, cos_x, -sin_x],
            [0, sin_x, cos_x]
        ])
        
        Ry = np.array([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ])
        
        Rz = np.array([
            [cos_z, -sin_z, 0],
            [sin_z, cos_z, 0],
            [0, 0, 1]
        ])
        
        return Rz @ Ry @ Rx
    
    @property
    def forward(self) -> np.ndarray:
        rotation_matrix = self._rotation_matrix()
        return -rotation_matrix[:, 2]
    
    @property
    def right(self) -> np.ndarray:
        rotation_matrix = self._rotation_matrix()
        return rotation_matrix[:, 0]
    
    @property
    def up(self) -> np.ndarray:
        rotation_matrix = self._rotation_matrix()
        return rotation_matrix[:, 1]
    
    def axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rotation_matrix = self._rotation_matrix()
        right = rotation_matrix[:, 0]
        up = rotation_matrix[:, 1]
        forward = -rotation_matrix[:, 2]
        return right, up, forward
    
    def look_at(self, target: np.ndarray, world_up: np.ndarray = np.array((0.0, 1.0, 0.0))):
        forward = target - self.position
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        rotation_matrix = np.column_stack([right, up, -forward])
        
        sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = 0
        
        self.rotation = np.degrees(np.array([x, y, z]))