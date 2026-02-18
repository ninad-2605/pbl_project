"""
Tier 4.4: Dynamic Packet Generator.

Generates variable-rate CSI packets for training with diverse temporal profiles.
Simulates realistic WiFi packet arrival patterns including:
- Normal operation (10-200Hz)
- High-speed capture (1000Hz)
- Chaos scenarios (CSMA/CA congestion, microwave interference gaps)
- Bursty traffic patterns
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from enum import Enum


class PacketProfile(Enum):
    """Predefined packet arrival profiles."""
    NORMAL_10HZ = "normal_10hz"       # Consumer WiFi card
    NORMAL_30HZ = "normal_30hz"       # Standard monitoring
    HIGH_SPEED_100HZ = "high_100hz"   # Intel AX cards
    ULTRA_1000HZ = "ultra_1000hz"     # Research-grade
    
    # Chaos Profiles
    CHAOS_CSMA = "chaos_csma"         # CSMA/CA congestion (gappy)
    CHAOS_MICROWAVE = "chaos_mw"      # Microwave interference (60Hz gaps)
    CHAOS_BLUETOOTH = "chaos_ble"     # BLE hopping interference
    CHAOS_CONGESTED = "chaos_congested"  # Multiple sources
    
    # Dynamic Profiles
    ADAPTIVE = "adaptive"             # Dynamically varies based on motion


@dataclass
class PacketConfig:
    """Configuration for packet generation."""
    base_rate_hz: float = 30.0
    rate_variance: float = 0.1       # ±10% rate variance
    dropout_prob: float = 0.0        # Packet loss probability
    burst_prob: float = 0.0          # Burst arrival probability  
    burst_size: Tuple[int, int] = (2, 5)  # Burst size range
    gap_prob: float = 0.0            # Gap (multi-packet loss) probability
    gap_duration_ms: Tuple[float, float] = (10.0, 100.0)  # Gap duration range


class DynamicPacketGenerator(nn.Module):
    """
    Generates variable-rate CSI packet sequences.
    
    Key Capabilities:
    1. Variable packet rates (1Hz - 1000Hz)
    2. Realistic packet timing with jitter
    3. Chaos scenarios (CSMA, interference, congestion)
    4. Learnable packet rate prediction (for adaptive systems)
    """
    
    # Rate limits for various profiles
    PROFILE_CONFIGS = {
        PacketProfile.NORMAL_10HZ: PacketConfig(base_rate_hz=10, rate_variance=0.1),
        PacketProfile.NORMAL_30HZ: PacketConfig(base_rate_hz=30, rate_variance=0.1),
        PacketProfile.HIGH_SPEED_100HZ: PacketConfig(base_rate_hz=100, rate_variance=0.05),
        PacketProfile.ULTRA_1000HZ: PacketConfig(base_rate_hz=1000, rate_variance=0.02),
        
        PacketProfile.CHAOS_CSMA: PacketConfig(
            base_rate_hz=30, rate_variance=0.3,
            dropout_prob=0.15, gap_prob=0.1, 
            gap_duration_ms=(20.0, 200.0)
        ),
        PacketProfile.CHAOS_MICROWAVE: PacketConfig(
            base_rate_hz=30, rate_variance=0.2,
            gap_prob=0.3, gap_duration_ms=(16.67, 16.67)  # 60Hz = 16.67ms
        ),
        PacketProfile.CHAOS_BLUETOOTH: PacketConfig(
            base_rate_hz=30, rate_variance=0.2,
            dropout_prob=0.05, burst_prob=0.1
        ),
        PacketProfile.CHAOS_CONGESTED: PacketConfig(
            base_rate_hz=20, rate_variance=0.5,
            dropout_prob=0.2, gap_prob=0.2, burst_prob=0.15,
            gap_duration_ms=(50.0, 500.0)
        ),
    }
    
    def __init__(self, device: str = 'cuda', num_subcarriers: int = 52):
        super().__init__()
        self.device = torch.device(device)
        self.num_subcarriers = num_subcarriers
        
        # Learnable rate predictor (for adaptive profile)
        self.rate_predictor = nn.Sequential(
            nn.Linear(128, 64),  # From latent
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Outputs 0-1, scaled to rate range
        ).to(self.device)
        
        # Rate bounds for adaptive mode
        self.min_rate_hz = 5.0
        self.max_rate_hz = 200.0
        
        print(f"[PacketGen] Dynamic Packet Generator Initialized")
        print(f"   - Subcarriers: {num_subcarriers}")
        print(f"   - Available Profiles: {[p.value for p in PacketProfile]}")
    
    def get_config(self, profile: PacketProfile) -> PacketConfig:
        """Get configuration for a profile."""
        return self.PROFILE_CONFIGS.get(profile, PacketConfig())
    
    def generate_timestamps(
        self, 
        duration_sec: float, 
        config: PacketConfig,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate realistic packet arrival timestamps.
        
        Args:
            duration_sec: Total duration in seconds
            config: Packet configuration
            seed: Random seed for reproducibility
            
        Returns:
            timestamps: [N] tensor of timestamps in seconds
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        timestamps = []
        t = 0.0
        
        # Base inter-packet interval
        base_interval = 1.0 / config.base_rate_hz
        
        while t < duration_sec:
            # Apply rate variance (jitter)
            jitter = 1.0 + (np.random.randn() * config.rate_variance)
            interval = base_interval * max(0.1, jitter)
            
            # Check for gap
            if np.random.rand() < config.gap_prob:
                gap_ms = np.random.uniform(*config.gap_duration_ms)
                interval = gap_ms / 1000.0
            
            # Check for burst
            elif np.random.rand() < config.burst_prob:
                burst_count = np.random.randint(*config.burst_size)
                for _ in range(burst_count):
                    if t < duration_sec:
                        timestamps.append(t)
                        t += interval * 0.1  # Rapid bursts
                continue
            
            # Check for dropout
            if np.random.rand() < config.dropout_prob:
                t += interval
                continue  # Skip this packet
            
            timestamps.append(t)
            t += interval
        
        return torch.tensor(timestamps, device=self.device)
    
    def generate_packet_sequence(
        self,
        csi_source: torch.Tensor,  # [T_source, Subcarriers] continuous CSI
        profile: PacketProfile = PacketProfile.NORMAL_30HZ,
        duration_sec: Optional[float] = None,
        target_packets: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Sample packets from continuous CSI source with realistic timing.
        
        Args:
            csi_source: [T_source, S] or [T_source, S, 2] continuous CSI
            profile: Packet arrival profile
            duration_sec: Override duration
            target_packets: Override to generate exact number (for training)
            
        Returns:
            packets: [N_packets, Subcarriers] sampled CSI
            timestamps: [N_packets] arrival times
            stats: Dictionary with generation statistics
        """
        config = self.get_config(profile) if isinstance(profile, PacketProfile) else profile
        
        T_source = csi_source.shape[0]
        source_duration = T_source / 1000.0  # Assume 1kHz source
        
        if target_packets is not None:
            # Generate exact number of packets (for fixed-size training)
            adjusted_rate = target_packets / source_duration
            config = PacketConfig(
                base_rate_hz=adjusted_rate,
                rate_variance=config.rate_variance,
                dropout_prob=config.dropout_prob
            )
        
        actual_duration = duration_sec if duration_sec else source_duration
        
        # Generate timestamps
        timestamps = self.generate_timestamps(actual_duration, config)
        
        # Sample CSI at timestamps
        # Map timestamps to source indices
        source_indices = (timestamps * 1000).long().clamp(0, T_source - 1)
        
        packets = csi_source[source_indices]
        
        # Statistics
        stats = {
            'profile': profile.value if isinstance(profile, PacketProfile) else 'custom',
            'num_packets': len(timestamps),
            'duration_sec': actual_duration,
            'effective_rate_hz': len(timestamps) / actual_duration if actual_duration > 0 else 0,
            'gaps_detected': (torch.diff(timestamps) > 0.1).sum().item(),
            'bursts_detected': (torch.diff(timestamps) < 0.01).sum().item()
        }
        
        return packets, timestamps, stats
    
    def adaptive_rate_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Predict optimal packet rate from model latent space.
        
        When motion is detected, we want higher rates.
        When scene is static, we can reduce rates to save power.
        
        Args:
            latent: [B, 128] latent code from model
            
        Returns:
            rate_hz: [B, 1] predicted optimal rate
        """
        rate_normalized = self.rate_predictor(latent)
        rate_hz = self.min_rate_hz + rate_normalized * (self.max_rate_hz - self.min_rate_hz)
        return rate_hz
    
    def resample_to_fixed_rate(
        self,
        packets: torch.Tensor,
        timestamps: torch.Tensor,
        target_rate_hz: float,
        duration_sec: float
    ) -> torch.Tensor:
        """
        Resample variable-rate packets to fixed output rate.
        Uses linear interpolation for missing samples.
        
        Args:
            packets: [N, S] variable-rate packets
            timestamps: [N] arrival times
            target_rate_hz: Desired output rate
            duration_sec: Total duration
            
        Returns:
            resampled: [T_out, S] fixed-rate CSI
        """
        T_out = int(duration_sec * target_rate_hz)
        S = packets.shape[1]
        
        resampled = torch.zeros(T_out, S, device=self.device, dtype=packets.dtype)
        
        # Generate target timestamps
        target_times = torch.linspace(0, duration_sec, T_out, device=self.device)
        
        # For each target time, find nearest packets and interpolate
        for i, t in enumerate(target_times):
            # Find bracketing timestamps
            mask_before = timestamps <= t
            mask_after = timestamps > t
            
            if mask_before.any() and mask_after.any():
                # Interpolate
                idx_before = mask_before.nonzero()[-1].item()
                idx_after = mask_after.nonzero()[0].item()
                
                t_before = timestamps[idx_before]
                t_after = timestamps[idx_after]
                
                alpha = (t - t_before) / (t_after - t_before + 1e-9)
                resampled[i] = (1 - alpha) * packets[idx_before] + alpha * packets[idx_after]
            elif mask_before.any():
                resampled[i] = packets[mask_before.nonzero()[-1].item()]
            elif mask_after.any():
                resampled[i] = packets[mask_after.nonzero()[0].item()]
        
        return resampled
    
    def forward(
        self,
        csi_source: torch.Tensor,
        profile: PacketProfile = PacketProfile.ADAPTIVE,
        latent: Optional[torch.Tensor] = None,
        output_rate_hz: float = 30.0
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Full pipeline: Generate packets and resample to fixed rate.
        
        Args:
            csi_source: [T, S] continuous CSI
            profile: Packet profile (or ADAPTIVE for latent-based)
            latent: [B, 128] latent code (for ADAPTIVE)
            output_rate_hz: Fixed output rate
            
        Returns:
            output_csi: [T_out, S] resampled CSI
            stats: Generation statistics
        """
        duration = csi_source.shape[0] / 1000.0
        
        if profile == PacketProfile.ADAPTIVE and latent is not None:
            # Dynamic rate based on latent
            predicted_rate = self.adaptive_rate_from_latent(latent).mean().item()
            config = PacketConfig(
                base_rate_hz=predicted_rate,
                rate_variance=0.1
            )
            packets, timestamps, stats = self.generate_packet_sequence(
                csi_source, config, duration
            )
            stats['predicted_rate_hz'] = predicted_rate
        else:
            packets, timestamps, stats = self.generate_packet_sequence(
                csi_source, profile, duration
            )
        
        # Resample to fixed output rate
        output = self.resample_to_fixed_rate(
            packets, timestamps, output_rate_hz, duration
        )
        
        stats['output_rate_hz'] = output_rate_hz
        stats['output_frames'] = output.shape[0]
        
        return output, stats


# Convenience functions
def get_chaos_profile(scenario: str) -> PacketProfile:
    """Map scenario name to chaos profile."""
    mapping = {
        'microwave': PacketProfile.CHAOS_MICROWAVE,
        'bluetooth': PacketProfile.CHAOS_BLUETOOTH,
        'ble': PacketProfile.CHAOS_BLUETOOTH,
        'csma': PacketProfile.CHAOS_CSMA,
        'congested': PacketProfile.CHAOS_CONGESTED,
        'crowded': PacketProfile.CHAOS_CONGESTED,
    }
    return mapping.get(scenario.lower(), PacketProfile.NORMAL_30HZ)


def estimate_model_capacity() -> Dict:
    """
    Estimate the model's packet ingestion capacity.
    
    Based on analysis of Zone1Thalamus and HeimdallInfinity:
    - Input: [B, C, F, T] where T is temporal dimension
    - STFT window: 256 samples at variable hop
    - LNN/Mamba can handle ~1000 timesteps before gradient degradation
    """
    return {
        'min_packets_per_window': 1,
        'max_packets_per_window': 1000,
        'optimal_range': (30, 200),
        'stft_window_size': 256,
        'stft_hop_sizes': [32, 64, 128, 256],
        'supported_rates_hz': [10, 30, 60, 100, 200, 500, 1000],
        'notes': [
            "LNN/Mamba handles variable-length sequences natively",
            "STFT window acts as temporal pooling",
            "Higher rates increase memory but improve temporal resolution",
            "Chaos scenarios may reduce effective rate by 20-50%"
        ]
    }
