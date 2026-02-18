"""
Tier 2: The World Graph.
Manages the simulation state: Entities, Geometry, and Spatial Partitioning.
"""

import torch
import numpy as np
from typing import List, Dict, Optional
import traceback
from dataclasses import dataclass, field
from src.core.paths import PathManager
from .materials import DielectricDatabase, MaterialProp
# Lazy import to avoid circular dependency if possible, or just import
# from src.engine.human import HumanEntity # Circular? World uses Entity, Human uses Entity.
# World uses Human. Human uses World? No. Human uses Entity.
# Should be fine.


@dataclass
class Transform:
    pos: torch.Tensor # [3]
    rot: torch.Tensor # [3, 3] Matrix or Quaternion
    scale: torch.Tensor # [3]

    @staticmethod
    def identity(device='cpu'):
        return Transform(
            pos=torch.zeros(3, device=device),
            rot=torch.eye(3, device=device),
            scale=torch.ones(3, device=device)
        )

class Entity:
    """Base Object in the Simulation."""
    def __init__(self, name: str, material: str, device='cuda'):
        self.name = name
        self.id = id(self)
        self.device = device
        self.transform = Transform.identity(device)
        self.material_name = material
        self.mesh_v: Optional[torch.Tensor] = None # [V, 3] Vertices
        self.mesh_f: Optional[torch.Tensor] = None # [F, 3] Faces
        
        # Physics State
        self.velocity = torch.zeros(3, device=device)
        self.is_static = True

    def set_mesh(self, vertices: torch.Tensor, faces: torch.Tensor):
        self.mesh_v = vertices.to(self.device)
        self.mesh_f = faces.to(self.device)
    
    def update(self, dt: float):
        if not self.is_static:
            self.transform.pos += self.velocity * dt

class World:
    """
    The Container of Existence.
    Manages the Scene Graph and Global State.
    """
    def __init__(self, device='cuda'):
        self.device = device
        self.entities: Dict[int, Entity] = {}
        self.humans: List['HumanEntity'] = [] # Optimized list for batch updates
        self.materials = DielectricDatabase()
        
        # Spatial Partitioning (BVH) could go here
        # For now, simplistic list
    
    def add(self, entity: Entity):
        self.entities[entity.id] = entity
        
    def create_box(self, name: str, material: str, pos: List[float], size: List[float]):
        """Procedurally create a box entity."""
        e = Entity(name, material, self.device)
        e.transform.pos = torch.tensor(pos, device=self.device)
        e.transform.scale = torch.tensor(size, device=self.device)
        
        # Basic Cube Mesh (Unit size centered at 0)
        # ... logic to generate cube verts/faces ...
        # (Placeholder for brevity, need a primitive generator utils)
        return e

    def spawn_probabilistic_router(self, room_dim: List[float] = [4.0, 4.0, 3.0]) -> Entity:
        """
        Spawns a Router in a 'Real World' location (Simulates bad placement).
        
        Statistics (Approx):
        - 50% TV Stand / Table (Low, near wall)
        - 20% Floor / Corner (Worst case)
        - 20% Ceiling / High Shelf (Ideal-ish)
        - 10% Hidden (Inside Cabinet - High Attenuation)
        """
        router = Entity("Router", "plastic", self.device)
        w, d, h = room_dim
        
        choice = np.random.rand()
        pos = np.zeros(3)
        
        if choice < 0.5:
            # Table/Stand (H = 0.5m to 0.8m, near wall)
            pos[0] = np.random.uniform(0, w)
            pos[1] = 0.1 # Near wall Y=0
            pos[2] = np.random.uniform(0.5, 0.8)
        elif choice < 0.7:
            # Floor/Corner (H = 0.1m)
            pos[0] = 0.1
            pos[1] = 0.1
            pos[2] = 0.1
        elif choice < 0.9:
            # High (H = 2.5m, Central)
            pos[0] = np.random.uniform(0, w)
            pos[1] = np.random.uniform(0, d)
            pos[2] = np.random.uniform(2.0, h-0.2)
        else:
            # Hidden (Inside Cabinet)
            # We don't spawn the cabinet mesh yet, but we mark it.
            # Effectively just a bad location + maybe a "Shroud" dielectric later.
            pos[0] = w - 0.5
            pos[1] = d - 0.5
            pos[2] = 0.5
            
        router.transform.pos = torch.tensor(pos, dtype=torch.float32, device=self.device)
        self.add(router)
        return router

    def create_dynamic_human(self, npz_path: str, model_path: str = None, segm_path: str = None, gender: str = 'neutral'):
        """
        Spawns a SMPL-X driven Human Entity.
        """
        # Default Paths (Cleanup for Portability)
        if model_path is None:
            model_path = str(PathManager.get_dataset_root("smplx"))
            
        if segm_path is None:
            segm_path = str(PathManager.get_dataset_root("smplx") / "smplx_parts_segm.pkl")
        
        # Dynamic import to avoid circular top-level
        from .human import HumanEntity
        
        # Check defaults
        if not npz_path:
             npz_path = "data/test_action_amass.npz"
             
        # Try-Catch for production safety
        try:
            h_ent = HumanEntity(
                "HumanSubject", 
                npz_path=npz_path, 
                model_path=model_path,
                gender=gender,
                device=self.device
            )
            
            # Load Segmentation
            h_ent.load_segmentation(segm_path)
            
            self.add(h_ent)
            self.humans.append(h_ent) # Add to optimized list
            return h_ent
        except Exception as e:
            print(f"[World] Failed to spawn SMPL Human: {e}. Fallback to Box.")
            traceback.print_exc()
            proxy = self.create_human_proxy()
            self.add(proxy)
            self.humans.append(proxy)
            return proxy

    def spawn_multi_humans(self, configs: List[Dict]):
        """
        Spawns multiple humans from a config list.
        Args:
            configs: List of dicts {amass_path, pos, gender}
        """
        print(f"[World] Spawning {len(configs)} Humans...")
        for i, cfg in enumerate(configs):
            self.create_dynamic_human(
                npz_path=cfg['amass_path'],
                gender=cfg.get('gender', 'neutral'),
            )
            # Set Position (HumanEntity needs direct transform access)
            if 'pos' in cfg:
                # Assuming HumanEntity inherits Entity and has transform
                self.humans[-1].transform.pos = torch.tensor(cfg['pos'], device=self.device)
            print(f"   + Human {i} ({cfg.get('gender')}) at {cfg.get('pos')}")

    def create_human_proxy(self):
        """
        Spawns a static 1.7m x 0.5m x 0.3m Box as a Human Proxy.
        (Fallback if SMPL missing).
        """
        h_ent = Entity("HumanProxy", "muscle", self.device)
        h_ent.transform.pos = torch.tensor([2.0, 2.0, 0.0], device=self.device) # Center of room
        
        # Create Mesh (Box)
        # 8 vertices, 12 triangles
        
        # Dimensions
        w, d, h = 0.5, 0.3, 1.7
        
        # Vertices (Centered at base)
        # x, y, z
        v = torch.tensor([
            [-w/2, -d/2, 0], [w/2, -d/2, 0], [w/2, d/2, 0], [-w/2, d/2, 0], # Bottom
            [-w/2, -d/2, h], [w/2, -d/2, h], [w/2, d/2, h], [-w/2, d/2, h]  # Top
        ], dtype=torch.float32, device=self.device)
        
        # Faces
        f = torch.tensor([
            [0, 1, 2], [0, 2, 3], # Bottom (Normal Down) - actually likely invisible
            [4, 5, 6], [4, 6, 7], # Top
            [0, 1, 5], [0, 5, 4], # Front
            [1, 2, 6], [1, 6, 5], # Right
            [2, 3, 7], [2, 7, 6], # Back
            [3, 0, 4], [3, 4, 7]  # Left
        ], dtype=torch.long, device=self.device)
        
        h_ent.set_mesh(v, f)
        h_ent.rest_mesh = v.clone() # Store for animation
        
        self.add(h_ent)
        return h_ent

    def update(self, t: float):
        """Global update loop (Animation & Physics integration)."""
        # Animate Entities (Breathing / Micro-motion)
        for e in self.entities.values():
            e.update(0.033) # Kinematics
            
            if e.name == "HumanProxy" and hasattr(e, 'rest_mesh'):
                # Breathing Animation
                # Expand Chest (Z > 1.0) by sin(t)
                # Simple vertex displacement
                
                cycle = np.sin(2.0 * np.pi * 0.3 * t) # 0.3 Hz breathing
                scale_factor = 1.0 + 0.05 * cycle # 5% expansion
                
                # Deform
                v = e.rest_mesh.clone()
                chest_mask = v[:, 2] > 0.9 # Upper body
                
                # Expand XY
                v[chest_mask, 0] *= scale_factor
                v[chest_mask, 1] *= scale_factor
                
                e.mesh_v = v # Update mesh
                
    def get_all_meshes(self):
        """Aggregate all meshes for the Ray Tracer."""
        all_v = []
        all_f = []
        
        offset = 0
        for e in self.entities.values():
            if e.mesh_v is not None and e.mesh_f is not None:
                # Apply Transform
                v = e.mesh_v * e.transform.scale
                if e.transform.rot.numel() == 9: # Matrix
                    v = v @ e.transform.rot.T
                
                v = v + e.transform.pos
                
                all_v.append(v)
                all_f.append(e.mesh_f + offset)
                
                offset += v.shape[0]
                
        if not all_v:
            return None, None
            
        return torch.cat(all_v), torch.cat(all_f)

    def load_scene(self, obj_path: str):
        """
        Loads a static environment from OBJ/GLTF using trimesh.
        Automatically maps materials to DielectricDatabase.
        """
        try:
            import trimesh
        except ImportError:
            print("[World] trimesh not installed. Cannot load scene.")
            return

        print(f"[World] Loading Scene: {obj_path}")
        scene = trimesh.load(obj_path, process=False) # process=False keeps structure
        
        # Normalize Scene (optional, maybe User wants specific scale)
        # For now, assume User provides Metric scale.
        
        geometries = []
        if isinstance(scene, trimesh.Scene):
            for name, geom in scene.geometry.items():
                geometries.append((name, geom))
        else:
            # Single mesh
            geometries.append(("Environment", scene))
            
        count = 0
        for name, geom in geometries:
            # Skip empty
            if not isinstance(geom, trimesh.Trimesh):
                continue
                
            # Determine Material from Name
            mat_name = "concrete" # Default
            lower_name = name.lower()
            
            if "wood" in lower_name or "floor" in lower_name:
                mat_name = "wood"
            elif "metal" in lower_name or "frame" in lower_name:
                mat_name = "metal"
            elif "glass" in lower_name or "window" in lower_name:
                mat_name = "glass"
            elif "drywall" in lower_name or "ceiling" in lower_name:
                mat_name = "drywall"
            elif "fabric" in lower_name or "couch" in lower_name:
                mat_name = "wood" # Approximation
            
            # Convert to Torch
            # Trimesh uses (N, 3) float64 usually
            v = torch.from_numpy(geom.vertices).float().to(self.device)
            f = torch.from_numpy(geom.faces).long().to(self.device)
            
            # Create Entity
            ent = Entity(name, mat_name, self.device)
            ent.set_mesh(v, f)
            
            # Apply Transform if in Scene graph?
            # Trimesh usually applies transforms to geometry in simple dump, 
            # but if we iterate scene.geometry, they are in local space?
            # scene.dump() or scene.to_mesh() concatenates.
            # If we want separate entities, we need to handle the graph graph.
            # Simplified: Use scene.dump(concatenate=True) for one big mesh?
            # No, we want material separation.
            # For now, let's assumes scene.geometry is flattened or we ignore hierarchy transforms for simple objs.
            # Proper way: iterate scene.graph nodes.
            # Start simple: Most BIM OBJs are just a bag of named meshes.
            
            self.add(ent)
            count += 1
            
        print(f"[World] Imported {count} meshes from scene.")
