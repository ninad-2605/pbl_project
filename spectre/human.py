"""
The Sovereign Human Entity.
Wraps SMPL-X Body Model and AMASS Motion Loader into a standard Entity.
"""

import torch
import numpy as np
from typing import Optional, Tuple
from pathlib import Path
import os

from src.data_generator.engine.world import Entity
from src.data_generator.motion.amass_loader import AMASSLoader, SMPLAdapter, AMASSWalker

class HumanEntity(Entity):
    """
    A dynamic human entity driven by AMASS motion data.
    Updates mesh and skeleton every frame.
    """
    def __init__(self, name: str,
                 npz_path: str = "datasets/amass/BMLrub/rub001/0005_normal_walk1_stageii.npz",
                 model_path: str = None,
                 gender: str = 'neutral',
                 device: str = 'cuda'):
        super().__init__(name, "human_skin", device)
        
        self.device = device
        
        # Load Motion
        if not os.path.exists(npz_path):
             # Try absolute path from config if relative fails
             if not os.path.isabs(npz_path):
                  try:
                      from src.core.paths import PathManager
                      potential_path = PathManager.get_dataset_root("amass") / npz_path.replace("datasets/amass/", "")
                      if potential_path.exists():
                           npz_path = str(potential_path)
                  except:
                      pass
                      
        # Model Path Logic
        if model_path is None:
             try:
                 from src.core.paths import PathManager
                 model_path = str(PathManager.get_dataset_root("smplx"))
             except:
                 model_path = "data/smplx_models"
        
        self.loader = AMASSLoader(npz_path, device=device)

        self.loader = AMASSLoader(npz_path, device=device)
        self.adapter = SMPLAdapter(model_path, gender=gender, device=device)
        self.walker = AMASSWalker(self.loader, self.adapter, loop=True)
        
        # State
        self.skeleton: Optional[torch.Tensor] = None # [52, 3]
        
        # Initial Pose
        self.update(0.0)
        
    def update(self, dt: float):
        """
        Advance animation by dt.
        Updates self.mesh_v, self.mesh_f, and self.skeleton.
        """
        # AMASSWalker returns (skeleton_52, vertices)
        # It handles frame advancement internally based on FPS
        
        # Note: AMASSWalker.update(dt) updates frame and returns state
        # We need to ensure dt is passed correctly (it expects seconds?)
        # Let's check amass_loader.py: update(dt) -> current_frame += dt * fps. Yes.
        
        skeleton, vertices = self.walker.update(dt)
        
        # Update Mesh (Vertices only, faces are constant usually)
        # But SMPLAdapter returns faces too.
        # We should cache faces if they don't change.
        # walker.update returns vertices. Adapter.forward returns faces.
        # Let's get faces from adapter once.
        if self.mesh_f is None:
             # We need to trigger a forward pass to get faces? 
             # Or just access self.adapter.model.faces if valid?
             # safe way: call adapter once.
             # Actually AMASSWalker.update calls adapter.forward.
             # But it only returns vertices. 
             # I should check if I need to modify AMASSWalker or just grab faces from adapter.
             
             # If using Mock, logic differs.
             # Let's assume consistent face topology.
             # We can grab it from a test forward pass or just assume static.
             pass

        self.mesh_v = vertices
        
        # To get faces, we can look at the adapter's last output or just run a dummy 
        # forward if needed at init. 
        # But wait, AMASSWalker.update returns (sov, vert).
        # We need faces for Ray Tracing. 
        # I'll create a helper or just modify walker to expose faces.
        # For now, let's assume standard SMPL faces.
        
        # HACK: If mesh_f is None, try to get it from adapter if model exists
        if self.mesh_f is None:
             if self.adapter.model:
                 faces = self.adapter.model.faces
                 self.mesh_f = torch.tensor(faces.astype(np.int64), device=self.device)
             else:
                 # Mock faces
                 self.mesh_f = torch.zeros((100, 3), dtype=torch.long, device=self.device)

        # Update Skeleton for HDF5 persistence
        self.skeleton = skeleton
        
        # Apply Transform? 
        # AMASS data is usually global.
        # If we want to move the human around the room (e.g. to [2,2,0]),
        # we can apply self.transform.pos to mesh_v and skeleton.
        # But AMASS has its own global trajectory.
        # We should probably reset AMASS root to (0,0,0) and use self.transform.
        # Logic for that is complex (root subtraction).
        # For now, let's assume AMASS global is "Good Enough" or we just override root.
        
        # Centering Logic (Optional):
        # center = skeleton[0] # Root
        # shift = self.transform.pos - center
        # self.mesh_v += shift
        # self.skeleton += shift
        
        
    def load_segmentation(self, pkl_path: str):
        """
        Loads SMPL-X part segmentation for Part-Specific Physics.
        """
        try:
            import pickle
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            
            # Data structure is likely {'segm': {PartName: [VertexIndices]}}
            if 'segm' in data:
                self.segmentation = data['segm']
                print(f"[HumanEntity] Loaded Segmentation: {len(self.segmentation)} parts.")
                
                
                if isinstance(self.segmentation, dict):
                    # Dict: PartName -> Indices
                    self.vertex_to_part = {}
                    for part, indices in self.segmentation.items():
                        if hasattr(indices, 'tolist'):
                            indices = indices.tolist()
                        for idx in indices:
                            self.vertex_to_part[idx] = part
                else:
                    # Array: VertexIndex -> PartID (Int)
                    # We need to map PartID to Name.
                    # Standard SMPL/SMPL-X Part Hierarchy (Approx 24 parts)
                    # 0: Pelvis, 1: L_Hip, 2: R_Hip, 3: Spine1, 4: L_Knee, 5: R_Knee, 
                    # 6: Spine2, 7: L_Ankle, 8: R_Ankle, 9: Spine3, 10: L_Foot, 11: R_Foot, 
                    # 12: Neck, 13: L_Collar, 14: R_Collar, 15: Head, 16: L_Shou, 17: R_Shou, 
                    # 18: L_Elbow, 19: R_Elbow, 20: L_Wrist, 21: R_Wrist, 22: L_Hand, 23: R_Hand
                    
                    part_names = [
                        'pelvis', 'leftHip', 'rightHip', 'spine1', 'leftKnee', 'rightKnee',
                        'spine2', 'leftAnkle', 'rightAnkle', 'spine3', 'leftFoot', 'rightFoot',
                        'neck', 'leftCollar', 'rightCollar', 'head', 'leftShoulder', 'rightShoulder',
                        'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist', 'leftHand', 'rightHand'
                    ]
                    
                    # Store array directly? Or map to dict?
                    # Array is faster. 
                    self.vertex_part_ids = self.segmentation # [V]
                    self.part_id_to_name = {i: name for i, name in enumerate(part_names)}
                    
                    print(f"[HumanEntity] Loaded Segmentation Array. Shape: {self.segmentation.shape}")
                        
            else:
                print("[HumanEntity] Segmentation file invalid (no 'segm' key).")
                self.segmentation = None
                        

                
        except Exception as e:
            print(f"[HumanEntity] Failed to load segmentation: {e}")
            self.segmentation = None

    def get_material_for_vertex(self, vertex_idx: int) -> str:
        """
        Returns the material name for a given vertex index based on segmentation.
        """
        part = "unknown"
        
        # Check Dict-based
        if hasattr(self, 'vertex_to_part'):
            part = self.vertex_to_part.get(vertex_idx, "unknown")
            
        # Check Array-based
        elif hasattr(self, 'vertex_part_ids'):
            if vertex_idx < len(self.vertex_part_ids):
                pid = self.vertex_part_ids[vertex_idx]
                part = self.part_id_to_name.get(pid, "unknown")
        
        # Material Mapping
        # SMPLX Part Names: head, neck, spine1, spine2, spine3, leftShoulder, rightShoulder...
        part_lower = part.lower()
        
        if part_lower in ['head', 'neck']:
            return "bone" # Skull/Cervical Spine
        elif 'hand' in part_lower or 'foot' in part_lower:
             return "skin_dry" # Thin
        elif 'spine' in part_lower or 'hip' in part_lower or 'pelvis' in part_lower:
             return "bone"
        elif 'arm' in part_lower or 'leg' in part_lower:
             return "muscle"
             
        return "muscle" # Default

    def get_segmented_meshes(self):
        """
        Returns list of (part_name, material_name, vertices, faces) for all body parts.
        Uses cached indices for efficiency.
        """
        if not hasattr(self, 'partitioned_faces'):
            # One-time initialization of partitions
            self._init_partitions()
            
        if not self.partitioned_faces:
             # Fallback
             return [("Body", "muscle", self.mesh_v, self.mesh_f)]
             
        results = []
        for part_name, (v_indices, sub_faces) in self.partitioned_faces.items():
            # Vertices for this part
            # v_indices is Tensor [N]
            # sub_faces is Tensor [M, 3] re-indexed to 0..N-1
            
            part_v = self.mesh_v[v_indices]
            
            # Material
            mat = "muscle"
            p_lower = part_name.lower()
            if p_lower in ['head', 'neck', 'spine1', 'spine2', 'spine3', 'pelvis']:
                mat = "bone"
            elif 'hand' in p_lower or 'foot' in p_lower:
                mat = "skin_dry"
            
            results.append((part_name, mat, part_v, sub_faces))
            
        return results

    def _init_partitions(self):
        """
        Precomputes face partitions based on segmentation.
        """
        if self.segmentation is None or self.mesh_v is None or self.mesh_f is None:
            self.partitioned_faces = {}
            return

        print("[HumanEntity] Partitioning Mesh for Physics...")
        # Self.vertex_part_ids [V] -> PartID
        # self.part_id_to_name
        
        # We need to assign Facess to Parts.
        # Rule: A face belongs to the Part of its first vertex (Simple)
        # Or mode of 3 vertices.
        
        f = self.mesh_f # [F, 3]
        v_pids = self.vertex_part_ids # [V] array
        
        # CPU processing for init
        f_cpu = f.cpu().numpy()
        # v_pids is numpy array
        
        # Face Part IDs = Mode of vertices
        # Optimized: just take vertex 0
        f_pids = v_pids[f_cpu[:, 0]] 
        
        self.partitioned_faces = {}
        
        # Group by Part ID
        unique_pids = np.unique(f_pids)
        
        for pid in unique_pids:
            part_name = self.part_id_to_name.get(pid, "Unknown")
            
            # Mask for faces
            mask_f = (f_pids == pid)
            sub_faces_orig_indices = f_cpu[mask_f] # [M, 3] referencing global indices
            
            # We need to compact vertices to create a valid sub-mesh (0..N-1)
            # Find unique vertices used by these faces
            unique_v_indices = np.unique(sub_faces_orig_indices)
            
            # Create Mapping Global -> Local
            # global_idx -> 0..K
            # We can use a partial lookup or just a dense map if indices are small?
            # Creating a map for 10k verts is fast.
            
            # Map: global_index -> local_index
            # We can use np.searchsorted if sorted
            # unique_v_indices is sorted by verify? np.unique returns sorted.
            
            # Remap faces
            # new_faces = searchsorted(unique_v_indices, sub_faces_orig_indices)
            
            remap = np.searchsorted(unique_v_indices, sub_faces_orig_indices)
            
            # Store as Tensors
            v_indices_torch = torch.from_numpy(unique_v_indices).to(self.device).long()
            sub_faces_torch = torch.from_numpy(remap).to(self.device).long()
            
            self.partitioned_faces[part_name] = (v_indices_torch, sub_faces_torch)
            
        print(f"[HumanEntity] Partitioned into {len(self.partitioned_faces)} sub-meshes.")
        """
        Generates procedural organ meshes positioned relative to the skeleton.
        Returns: List of (name, material, vertices, faces)
        """
        if self.skeleton is None:
            return []
            
        organs = []
        
        # Helper: Create Sphere/Box Primitive
        def create_sphere(radius, center, res=8):
             # Simple UV sphere
             phi = torch.linspace(0, np.pi, res, device=self.device)
             theta = torch.linspace(0, 2*np.pi, res, device=self.device)
             phi, theta = torch.meshgrid(phi, theta, indexing='ij')
             
             x = radius * torch.sin(phi) * torch.cos(theta)
             y = radius * torch.sin(phi) * torch.sin(theta)
             z = radius * torch.cos(phi)
             
             v = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1) + center
             
             # Faces (Grid topology)
             # ... simplified hull or just point cloud for voxelizer? 
             # Voxelizer needs faces for inside test.
             # Importing a primitive sphere mesh is better.
             # Or just use a Cube for now?
             return v, None # Faces TODO
        
        # Better Helper: Create Scaled Cube (Voxelizer handles cubes fine)
        def create_box(size, center, rot_mat=None):
            w, h, d = size
            v = torch.tensor([
                [-w, -h, -d], [w, -h, -d], [w, h, -d], [-w, h, -d],
                [-w, -h, d], [w, -h, d], [w, h, d], [-w, h, d]
            ], device=self.device) * 0.5
            
            if rot_mat is not None:
                v = v @ rot_mat.T
                
            v = v + center
            
            f = torch.tensor([
                [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
                [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]
            ], device=self.device, dtype=torch.long)
            
            return v, f
            
        # Sovereign Skeleton Mapping (Approx)
        # 1: Neck, 2: Head (Brain)
        # 5: L_Shou, 6: R_Shou (Chest/Lungs)
        # 12: R_Hip, 11: L_Hip (Abdomen)
        
        skel = self.skeleton
        
        # 1. BRAIN (Head Center)
        head_pos = skel[2] + torch.tensor([0.0, 0.0, 0.05], device=self.device) # Up bit
        v_brain, f_brain = create_box((0.14, 0.16, 0.14), head_pos)
        organs.append(('Brain', 'brain', v_brain, f_brain))
        
        # 2. HEART (Left Chest)
        # Between Neck(1) and Chest center.
        spine_top = skel[1] # Neck
        chest_center = (skel[5] + skel[6]) * 0.5
        heart_pos = chest_center + torch.tensor([0.04, -0.02, -0.05], device=self.device) # Left side
        v_heart, f_heart = create_box((0.10, 0.08, 0.12), heart_pos)
        organs.append(('Heart', 'heart', v_heart, f_heart))
        
        # 3. LUNGS (Left/Right)
        lung_size = (0.12, 0.15, 0.25)
        # Left Lung
        l_pos = chest_center + torch.tensor([0.08, 0.0, -0.1], device=self.device)
        v_ll, f_ll = create_box(lung_size, l_pos)
        organs.append(('Lung_L', 'lung_inf', v_ll, f_ll))
        
        # Right Lung
        r_pos = chest_center + torch.tensor([-0.08, 0.0, -0.1], device=self.device)
        v_rl, f_rl = create_box(lung_size, r_pos)
        organs.append(('Lung_R', 'lung_inf', v_rl, f_rl))
        
        return organs
        
