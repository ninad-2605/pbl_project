"""
Tier 4.3: The Noise Twin.
Learnable noise models that can be calibrated against real-world data.

Wraps HardwareWrapper and InterferenceEngine with nn.Parameters.
"""

import torch
import torch.nn as nn
import math

class LearnableNoiseTwin(nn.Module):
    """
    Differentiable Noise Model for Digital Twin Calibration.
    
    Learns:
    - Thermal noise floor (dBm)
    - IQ imbalance (amplitude and phase)
    - Interference power levels
    """
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = torch.device(device)
        
        # ===== Hardware Noise (Learnable) =====
        # Thermal Noise Floor (dBm) - stored in log space
        # Typical range: -100 to -80 dBm
        self.thermal_noise_dbm = nn.Parameter(torch.tensor(-95.0))
        
        # IQ Imbalance
        # Amplitude Error (linear scale, centered at 1.0)
        self.iq_amp_error = nn.Parameter(torch.tensor(0.05))
        # Phase Error (radians)
        self.iq_phase_error = nn.Parameter(torch.tensor(0.05))
        
        # ===== Interference Noise (Learnable) =====
        # Microwave Power (dBm)
        self.microwave_power_dbm = nn.Parameter(torch.tensor(-50.0))
        
        # Bluetooth Power (dBm)
        self.bluetooth_power_dbm = nn.Parameter(torch.tensor(-60.0))
        
        # Neighbor WiFi Load Factor (0-1)
        self.neighbor_load = nn.Parameter(torch.tensor(0.1))
        
        print(f"[NoiseTwin] Learnable Noise Model Initialized ({sum(p.numel() for p in self.parameters())} params)")
        
    def apply_hardware_noise(self, csi: torch.Tensor) -> torch.Tensor:
        """
        Applies learnable hardware impairments.
        """
        # 1. IQ Imbalance
        amp_scale = 1.0 + self.iq_amp_error
        phase_shift = self.iq_phase_error
        
        csi = csi * amp_scale * torch.exp(1j * phase_shift)
        
        # 2. Thermal Noise (AWGN)
        noise_power_linear = 10 ** (self.thermal_noise_dbm / 10.0)
        # Generate complex noise
        noise_real = torch.randn_like(csi.real) * torch.sqrt(noise_power_linear / 2)
        noise_imag = torch.randn_like(csi.imag) * torch.sqrt(noise_power_linear / 2)
        noise = torch.complex(noise_real, noise_imag)
        
        return csi + noise
        
    def apply_interference(self, csi: torch.Tensor, profile: dict) -> torch.Tensor:
        """
        Applies learnable interference.
        
        Args:
            csi: Complex CSI tensor
            profile: {'microwave': bool, 'bluetooth': bool, 'neighbors': int}
        """
        # 1. Microwave (if enabled)
        if profile.get('microwave', False):
            mw_power_linear = 10 ** (self.microwave_power_dbm / 10.0)
            # Simple additive noise model (60Hz modulated in real scenario)
            mw_noise = torch.randn_like(csi.real) * torch.sqrt(mw_power_linear)
            csi = csi + mw_noise
            
        # 2. Bluetooth (if enabled)
        if profile.get('bluetooth', False):
            bt_power_linear = 10 ** (self.bluetooth_power_dbm / 10.0)
            # Frequency hopping creates sparse spectral interference
            # Simplified: random subcarrier corruption
            hop_mask = torch.rand(csi.shape[-1], device=self.device) < 0.1  # 10% of subcarriers
            bt_noise = torch.randn_like(csi.real) * torch.sqrt(bt_power_linear)
            csi = csi + bt_noise * hop_mask
            
        # 3. Neighbor WiFi
        n_neighbors = profile.get('neighbors', 0)
        if n_neighbors > 0:
            # Co-channel interference scales with load
            cci_power = self.neighbor_load * n_neighbors * 1e-9
            cci_noise = torch.randn_like(csi.real) * torch.sqrt(cci_power)
            csi = csi + cci_noise
            
        return csi
        
    def forward(self, csi_clean: torch.Tensor, profile: dict = None) -> torch.Tensor:
        """
        Full noise application pipeline.
        """
        if profile is None:
            profile = {}
            
        csi = self.apply_hardware_noise(csi_clean)
        csi = self.apply_interference(csi, profile)
        
        return csi
