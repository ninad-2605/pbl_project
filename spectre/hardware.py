"""
The Dirty Layer: Hardware Emulation.
Ports 'DataSentry' and 'SensorDegradation' from legacy generators.
Ensures data robustness against real-world imperfections.
Integrates complete PHY Impairment Stack (Tier 19).
"""

import torch
import numpy as np
import logging

from src.core.utils.zero_copy_buffer import ZeroCopyBuffer
from src.physics.phy_impairments import PHYImpairmentWrapper, ImpairmentConfig

class HoloscanBypassIngest:
    """
    Tier 1: High-Speed User-Space Ingest.
    Bypasses OS kernel overhead by reading from a pinned ZeroCopyBuffer.
    Target: 10,000Hz stable signal capture.
    """
    def __init__(self, buffer_size: int = 1024 * 1024): # 1MB Buffer
        self.buffer = ZeroCopyBuffer(buffer_size)
        print(f"[HoloscanBypass] User-Space Ingest Live (Buffer: {buffer_size/1024} KB)")

    def capture_frame(self, shape, dtype=torch.float32):
        """Zero-copy capture into PyTorch."""
        return self.buffer.as_torch(shape, dtype)

class HardwareWrapper:
    def __init__(self, device='cuda', profile='random', bypass_mode=False):
        self.device = torch.device(device)
        self.profile = profile
        self.bypass_mode = bypass_mode
        self.ingest = None
        
        if self.bypass_mode:
            self.ingest = HoloscanBypassIngest()
        
        # Initialize PHY Stack based on Profile
        self.phy = self._init_phy_profile(profile)
        
        print(f"[Hardware] Initialized Hardware Wrapper (Profile: {profile}, Bypass: {bypass_mode})")
        if self.phy:
            print(f"   - PHY Stack: {self.phy.get_impairment_summary()}")

    def _init_phy_profile(self, profile: str) -> PHYImpairmentWrapper:
        """Configures the impairment stack based on requested profile."""
        if profile == 'ideal':
            # No impairments
            cfg = ImpairmentConfig(
                cfo_hz_range=(0,0), cfo_drift_rate=0,
                sfo_ppm_range=(0,0),
                iq_gain_db_range=(0,0), iq_phase_deg_range=(0,0),
                phase_noise_std=0,
                snr_db_range=(100, 100) # High SNR
            )
            # Disable quantization/AGC by setting them None in wrapper init if needed,
            # but Random factory uses config ranges.
            # Ideally we construct explicitly for 'ideal'.
            return PHYImpairmentWrapper(config=cfg) # Empty wrapper?
            
        elif profile == 'textbook':
            # Standard 3GPP Typical Values
            cfg = ImpairmentConfig(
                cfo_hz_range=(-100, 100),
                sfo_ppm_range=(-5, 5),
                iq_gain_db_range=(-0.2, 0.2),
                iq_phase_deg_range=(-1.0, 1.0),
                phase_noise_std=0.005,
                snr_db_range=(-60, -60)
            )
            return PHYImpairmentWrapper.random(cfg)
            
        elif profile == 'harsh':
            # Severe degradation (Low SNR, High Drift)
            return PHYImpairmentWrapper.from_snr_db(-85.0)
            
        elif profile == 'random':
            # Full randomized range
            return PHYImpairmentWrapper.random()
            
        else:
            return PHYImpairmentWrapper.random()

    def apply_degradation(self, csi_clean: torch.Tensor) -> torch.Tensor:
        """
        Applies 'Dirty RF' effects to clean synthetic CSI.
        Delegates to PHYImpairmentWrapper.
        """
        # If random profile was selected at init, we might want to randomize per-call?
        # The PHYImpairmentWrapper holds Specific instances (e.g. a specific CFO drift obj).
        # We usually want to vary parameters per Sample or per Batch?
        # If we stick to one wrapper instance, the drift states (random walk) are preserved, 
        # simulating a continuous session. This is GOOD for time-series.
        
        # However, if we want Domain Randomization across batches, we might reset.
        # For Sovereign Engine (Simulator), preserving state allows simulating "Drift".
        
        return self.phy.apply(csi_clean, sample_rate=20e6, num_subcarriers=52)

class DataSentry:
    """Validator for Data Integrity."""
    @staticmethod
    def check(csi: torch.Tensor) -> bool:
        if torch.isnan(csi).any() or torch.isinf(csi).any():
            return False
        if csi.abs().mean() < 1e-9: # Dead signal
            return False
        return True
