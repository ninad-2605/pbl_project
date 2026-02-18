"""
The Sovereign Orchestrator.
Master controller for the NVIDIA Sovereign Reference Implementation.
Coordinates World State, Physics Simulation, and Data Persistence.
"""

import torch
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

from src.data_generator.engine.world import World
# from src.data_generator.engine.physics_bridge import PhysicsBridge (Legacy)
from src.data_generator.engine.hardware import HardwareWrapper
from src.data_generator.engine.interference import InterferenceEngine
from src.physics.motion_synthesis import FrankensteinMotionEngine

class SovereignOrchestrator:
    def __init__(self, output_dir: str = "data/output", device: str = 'cuda', amass_path: str = None):
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Initialize World
        self.world = World(device=device)
        
        # Expanded Room: 10x8x3m (Warehouse / Large Office)
        room_dims = [10.0, 8.0, 3.0] 
        self.world.room_dim = np.array(room_dims)
        
        # Spawn Router (Access Point)
        self.router = self.world.spawn_probabilistic_router(room_dim=room_dims)
        print(f"   - Router spawned at: {self.router.transform.pos.tolist()}")
        
        # Spawn 5 Humans (Multi-Person Sovereign)
        # We try to use different AMASS files if available, otherwise reuse 'test_action'
        # with motion augmentation.
        
        if amass_path is None:
            # Try to resolve default via PathManager
            try:
                from src.core.paths import PathManager
                amass_root = PathManager.get_dataset_root("amass")
                amass_base = str(amass_root / "BMLrub/rub001/0005_normal_walk1_stageii.npz")
            except:
                 amass_base = "data/test_action_amass.npz"
        else:
            amass_base = amass_path

        # Verify existence
        if not Path(amass_base).exists():
            print(f"[Orchestrator] Warning: AMASS file {amass_base} not found. Using internal fallback.")
            amass_base = "" # Let World handle fallback to box or internal default

        human_configs = [
            {'amass_path': amass_base, 'pos': [2.0, 2.0, 0.0], 'gender': 'male'},
            {'amass_path': amass_base, 'pos': [8.0, 2.0, 0.0], 'gender': 'female'},
            {'amass_path': amass_base, 'pos': [2.0, 6.0, 0.0], 'gender': 'neutral'},
            {'amass_path': amass_base, 'pos': [8.0, 6.0, 0.0], 'gender': 'female'},
            {'amass_path': amass_base, 'pos': [5.0, 4.0, 0.0], 'gender': 'male'}, # Center
        ]
        
        self.world.spawn_multi_humans(human_configs)
        print(f"   - Spawned {len(self.world.humans)} Human Entities.")
        
        # 2. Motion Augmentation (Phase E)
        self._augment_human_motion()

        # 3. Initialize Physics Engines (Legacy Bridge Removed)
        self.physics = None 
        self.chaos = InterferenceEngine(device=device)
        self.hardware = HardwareWrapper(device=device, profile='random') # Random hardware defects
        
        print("[Orchestrator] System Online.")
        
    def _augment_human_motion(self):
        """
        Apply Frankenstein Motion Synthesis to differentiate identical clones.
        Adds Procedural Sway and Latent Noise.
        """
        augmenter = FrankensteinMotionEngine(device=self.device)
        print(f"[Orchestrator] Augmenting Motion for {len(self.world.humans)} humans...")
        
        for i, human in enumerate(self.world.humans):
            try:
                # Access Data loaded by AMASSLoader
                # pose_body: [T, 63] (flattened)
                raw_pose = human.loader.data['pose_body']
                T = raw_pose.shape[0]
                
                # Reshape to [1, T, 21, 3] for manipulation
                pose_3d = raw_pose.view(1, T, 21, 3) 
                
                # 1. Apply Sway (Idle variation)
                # Varies freq/amp per person
                freq = 0.2 + (i * 0.1) # 0.2, 0.3, 0.4...
                amp = 0.05 + (0.02 * (i % 3))
                
                # Note: 'inject_procedural_sway' targets indices [3,6,9] (Spine).
                # Our pose_3d starts at Joint 1 (L_Hip).
                # SMPL Spine1 is Joint 3. In 0-indexed pose_body, it is Index 2.
                # Frankenstein uses absolute SMPL indices. 
                # We need to map or modify Frankenstein. 
                # Easier hack: Shift Frankenstein indices or pass a padded tensor.
                
                # Let's simple-hack: Add minimal noise to 'pose_3d' indices 2, 5, 8
                # Manually here for safety vs Frankenstein implementation mismatch.
                
                # Manual Sway (Explicit Broadcasting)
                t_frames = torch.arange(T, device=self.device).float() / 30.0
                sway = (torch.sin(2.0 * np.pi * freq * t_frames) * amp).view(1, T, 1) # [1, T, 1]
                
                # Apply to Spine joints (Pitch)
                target_slice = pose_3d[:, :, [2, 5, 8], 0] # [1, T, 3]
                pose_3d[:, :, [2, 5, 8], 0] = target_slice + sway
                
                # 2. Latent Noise (Jitter)
                noise = torch.randn_like(pose_3d) * 0.01 
                pose_3d += noise
                
                # Write back
                human.loader.data['pose_body'] = pose_3d.view(T, 63)
                
                # Also randomize beta (Body Shape) slightly
                # betas: [16]
                if human.loader.data.get('betas') is not None:
                     human.loader.data['betas'] += torch.randn_like(human.loader.data['betas']) * 0.5
                
            except Exception as e:
                print(f"Warning: Failed to augment motion for human {i}: {e}")

    def run_simulation(self, total_time: float = 2.0, dt: float = 0.033):
        """
        Main Loop: Physics -> Chaos -> Hardware -> Storage
        """
        total_frames = int(total_time / dt)
        print(f"[Orchestrator] Starting Simulation ({total_time}s, {total_frames} frames)...")
        
        output_path = self.output_dir / "sovereign_data_v2.h5"
        
        with h5py.File(output_path, 'w') as f:
            # Datasets
            # CSI: [T, 52, 2] -> Just 1 stream (SISO) for simplicity or [T, 5, 52, 2] if MIMO/User segregated?
            # Standard: [T, Subcarriers, Complex]
            ds_csi = f.create_dataset("csi", (total_frames, 52), dtype='complex64')
            ds_time = f.create_dataset("timestamp", (total_frames,), dtype='f4')
            
            # Skeleton Shape: [T, 5, 52, 3] (5 People, 52 Joints, 3D)
            num_people = 5
            ds_skel = f.create_dataset("skeleton", (total_frames, num_people, 52, 3), dtype='f4')      
            
            for t in tqdm(range(total_frames)):
                sim_time = t * dt
                
                # 1. Update World State (Kinematics)
                self.world.update(dt)
                
                # 2. Physics Step (Ray Tracing + Body Physics)
                # Returns 'clean' CSI
                csi_clean, fields = self.physics.step(self.world, dt)
                
                # 3. Chaos Injection (Interference)
                # Add microwave/neighbor noise
                chaos_profile = {'microwave': (t % 100 < 50), 'neighbors': 2} # Dynamic profile
                
                # Convert Complex -> Real View [..., 2] for InterferenceEngine
                # csi_clean is [52] complex64
                csi_view = torch.view_as_real(csi_clean) # [52, 2]
                csi_input = csi_view.unsqueeze(0).unsqueeze(0) # [1, 1, 52, 2]
                
                csi_chaotic_view = self.chaos.apply_chaos(csi_input, chaos_profile)
                
                # Convert back to Complex
                if csi_chaotic_view.is_complex():
                     csi_chaotic = csi_chaotic_view.squeeze()
                else:
                     csi_chaotic = torch.view_as_complex(csi_chaotic_view).squeeze() # [52]
                
                # 4. Hardware Degradation (PHY Impairments)
                # Apply IQ, CFO, Phase Noise
                csi_dirty = self.hardware.apply_degradation(csi_chaotic)
                
                # 5. Save Data
                ds_csi[t] = csi_dirty.cpu().numpy()
                ds_time[t] = sim_time
                
                # Save Skeletons for all 5 Humans
                for i, human in enumerate(self.world.humans):
                    if i >= num_people: break 
                    if human.skeleton is not None:
                        # [52, 3]
                        skel_full = human.skeleton.detach().cpu().numpy()
                        ds_skel[t, i] = skel_full
                        
        print(f"[Orchestrator] Simulation Complete. Saved to {output_path}")
        return str(output_path)

if __name__ == "__main__":
    # Self-Test
    orch = SovereignOrchestrator()
    orch.run_simulation(total_time=0.1)
