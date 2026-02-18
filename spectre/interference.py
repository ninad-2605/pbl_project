"""
The Chaos Engine: Realistic RF Interference.
Unifies:
1. Microwave Ovens (Spectrum Chaos)
2. Bluetooth Hopping (Spectrum Chaos)
3. Neighbor WiFi Networks (Neighborhood)
4. Static Multipath (Multipath)
"""

import torch
import numpy as np
from src.physics.spectrum_chaos import InterferenceGenerator
from src.data_generator.scenarios.neighborhood import GhostNetwork
from src.physics.multipath import StaticMultipathGenerator

class InterferenceEngine:
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        
        # Sub-Engines
        self.jammer = InterferenceGenerator(device=device) # Microwave, BLE
        self.ghosts = GhostNetwork(device=device)          # Neighbors
        self.multipath = StaticMultipathGenerator(seed=42) # Static Room Reflections
        
        print("[Interference] Interference Engine Online: Microwave, BLE, Neighbors, Multipath Ready.")

    def apply_chaos(self, 
                    csi_clean: torch.Tensor, 
                    scenario_profile: dict) -> torch.Tensor:
        """
        Injects configured interference into the clean CSI.
        
        Args:
            csi_clean: [B, T, S, 2] Complex CSI
            scenario_profile: Dict with flags (e.g., 'microwave': True, 'neighbors': 3)
            
        Returns:
            csi_dirty: [B, T, S, 2]
        """
        csi = csi_clean.clone()
        
        # 1. Static Multipath (The Room)
        # Adds a DC component to Real/Imag parts
        # This simulates wall reflections that are constant over the window
        static_amp, static_phase = self.multipath.get_static_components()
        
        # Convert to Complex Adder
        # Static multipath is a scalar complex value added to all subcarriers (simplified DC offset)
        # In reality, it varies by frequency, but for "Ghost Rays" scalar is a good Tier 1 approx.
        static_complex = static_amp * np.exp(1j * static_phase)
        static_tensor = torch.tensor(static_complex, device=self.device, dtype=csi.dtype)
        
        csi = csi + static_tensor
        
        # 2. Microwave Oven (60Hz Pulse)
        if scenario_profile.get('microwave', False):
            csi = self.jammer.inject_microwave_blast(csi, power_dbm=-50.0)
            
        # 3. Bluetooth (Frequency Hopping)
        if scenario_profile.get('bluetooth', False):
            csi = self.jammer.inject_bluetooth_hop(csi)
            
        # 4. Neighbor WiFi (Collisions & Leakage)
        n_neighbors = scenario_profile.get('neighbors', 0)
        if n_neighbors > 0:
            # Co-Channel Interference (Collisions)
            csi = self.ghosts.inject_cci(csi, traffic_load=0.1 * n_neighbors)
            # Adjacent Channel Interference (Spectral Leakage)
            csi = self.ghosts.inject_aci(csi, num_neighbors=n_neighbors)
            
        return csi
