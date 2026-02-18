
import torch
import warp as wp
import numpy as np
from typing import Optional, Union, Tuple

from src.physics.sionna_clones.warp_radio_material import WarpRadioMaterial

@wp.kernel
def transform_points_kernel(
    src_points: wp.array(dtype=wp.vec3),
    dest_points: wp.array(dtype=wp.vec3),
    pos: wp.vec3,
    rot: wp.quat,
    scale: wp.vec3
):
    tid = wp.tid()
    p = src_points[tid]
    
    # 1. Scale
    p_s = wp.vec3(p[0]*scale[0], p[1]*scale[1], p[2]*scale[2])
    
    # 2. Rotate
    p_r = wp.quat_rotate(rot, p_s)
    
    # 3. Translate
    p_w = p_r + pos
    
    dest_points[tid] = p_w

class WarpSceneObject:
    """
    Clone of sionna.rt.SceneObject using NVIDIA Warp.
    Manages mesh geometry, transforms, and material assignment.
    """
    def __init__(
        self, 
        name: str, 
        vertices: torch.Tensor, 
        faces: torch.Tensor, 
        material: Optional[WarpRadioMaterial] = None,
        device: str = "cuda"
    ):
        self.name = name
        self.device = device
        self.material = material
        
        # 1. Geometry (Raw/Template)
        self.num_verts = vertices.shape[0]
        self.num_faces = faces.shape[0]
        
        # Store as Warp arrays
        # Ensure tensors are on the correct device for Warp
        self.raw_verts = wp.from_torch(vertices.to(self.device).to(torch.float32).contiguous(), dtype=wp.vec3)
        self.faces = wp.from_torch(faces.to(self.device).to(torch.int32).flatten().contiguous(), dtype=wp.int32)
        
        # Destination buffer for World Coordinates
        self.world_verts = wp.zeros_like(self.raw_verts)
        
        # 2. Transforms
        self.position = wp.vec3(0.0, 0.0, 0.0)
        self.orientation = wp.quat_identity()
        self.scale = wp.vec3(1.0, 1.0, 1.0)
        
        # 3. Initialize Physics State
        self.update_geometry()
        
        # 4. BVH (Warp Mesh)
        # Note: We use world_verts so ray tracing happens in world space directly
        self.mesh = wp.Mesh(self.world_verts, self.faces)
        
    def update_geometry(self):
        """
        Applies S -> R -> T transforms to raw vertices using Warp kernel.
        """
        wp.launch(
            kernel=transform_points_kernel,
            dim=self.num_verts,
            inputs=[
                self.raw_verts,
                self.world_verts,
                self.position,
                self.orientation,
                self.scale
            ],
            device=self.device
        )
        # Note: If mesh was already created, we use refit for performance.
        # Refit is ~10x-100x faster than rebuilding the BVH tree.
        if hasattr(self, 'mesh'):
            self.mesh.refit()
        else:
            # First-time build
            self.mesh = wp.Mesh(self.world_verts, self.faces)

    def set_position(self, x: float, y: float, z: float):
        self.position = wp.vec3(x, y, z)
        self.update_geometry()

    def set_scale(self, x: float, y: float, z: float):
        self.scale = wp.vec3(x, y, z)
        self.update_geometry()
        
    def set_orientation_quat(self, x: float, y: float, z: float, w: float):
        """
        Set orientation directly using a quaternion (x, y, z, w).
        """
        self.orientation = wp.quat(x, y, z, w)
        self.update_geometry()

    def look_at(self, target: Union[torch.Tensor, Tuple[float, float, float]]):
        """
        Points the object's X-axis towards the target.
        """
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        
        target_pos = np.array(target, dtype=np.float64)
        # wp.vec3 doesn't have .numpy() - extract components manually
        current_pos = np.array([float(self.position[0]), float(self.position[1]), float(self.position[2])], dtype=np.float64)
        
        # Direction vector
        forward = target_pos - current_pos
        norm = np.linalg.norm(forward)
        if norm < 1e-6:
            return # Degenerate case
            
        forward = forward / norm
        
        # Warp Quat from Forward vector (Assuming X is forward)
        # Default X is (1, 0, 0).
        # We need rotation R such that R * (1, 0, 0) = forward.
        # Axis = cross((1,0,0), forward)
        # Angle = acos(dot((1,0,0), forward))
        
        ref = np.array([1.0, 0.0, 0.0])
        axis = np.cross(ref, forward)
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm < 1e-6:
            # Parallel or anti-parallel
            if np.dot(ref, forward) < 0:
                # 180 degrees around Z
                self.orientation = wp.quat(0.0, 0.0, 1.0, 0.0)
            else:
                # Identity
                self.orientation = wp.quat_identity()
        else:
            axis = axis / axis_norm
            angle = np.arccos(np.clip(np.dot(ref, forward), -1.0, 1.0))
            
            # Quat from axis-angle: [x*sin(a/2), y*sin(a/2), z*sin(a/2), cos(a/2)]
            s = np.sin(angle / 2.0)
            c = np.cos(angle / 2.0)
            
            self.orientation = wp.quat(axis[0]*s, axis[1]*s, axis[2]*s, c)
            
        self.update_geometry()
        
    def Refit(self):
        self.mesh = wp.Mesh(self.world_verts, self.faces)

    def get_world_vertices(self) -> torch.Tensor:
        """
        Returns world vertices as torch tensor.
        """
        return wp.to_torch(self.world_verts)
