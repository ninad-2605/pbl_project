"""
Tier 4.5: Digital Twin Calibration Loop.

End-to-end calibration pipeline that optimizes the Digital Twin
(materials, noise, packet timing) against real/reference CSI data.

This is the final piece of the "Bi-Directional Truth" system.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class CalibrationConfig:
    """Configuration for calibration loop."""
    epochs: int = 100
    lr_materials: float = 1e-4
    lr_noise: float = 1e-3
    lr_packet: float = 1e-3
    
    # Loss weights
    weight_csi_amplitude: float = 1.0
    weight_csi_phase: float = 0.5
    weight_temporal_coherence: float = 0.3
    
    # Gradient stability
    grad_clip_norm: float = 1.0
    warmup_epochs: int = 10
    
    # Checkpointing
    save_every: int = 20
    eval_every: int = 10


class DigitalTwinCalibrator(nn.Module):
    """
    Full Digital Twin Calibration System.
    
    Combines:
    - Differentiable FDTD (physics)
    - Learnable Materials (epsilon_r)
    - Learnable Noise Twin (hardware impairments)
    - Dynamic Packet Generator (temporal modeling)
    
    Optimizes all parameters jointly to match observed CSI.
    """
    
    def __init__(self, orchestrator, config: CalibrationConfig = None):
        super().__init__()
        self.orchestrator = orchestrator
        self.config = config or CalibrationConfig()
        
        if orchestrator is None:
             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
             return

        self.device = orchestrator.device
        
        # Verify Tier 4 components are available
        assert orchestrator.learnable_materials is not None, "LearnableDielectricDatabase required"
        assert orchestrator.diff_fdtd is not None, "DiffFDTDLayer required"
        
        # Track calibration state
        self.calibration_history = {
            'loss': [],
            'loss_amplitude': [],
            'loss_phase': [],
            'epsilon_values': {},
            'noise_params': {}
        }
        
        print(f"[Calibrator] Digital Twin Calibrator Initialized")
        print(f"   - Epochs: {self.config.epochs}")
        print(f"   - LR (Materials): {self.config.lr_materials}")
        print(f"   - LR (Noise): {self.config.lr_noise}")
    
    def compute_csi_loss(
        self, 
        simulated_csi: torch.Tensor, 
        target_csi: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute calibration loss between simulated and target CSI.
        
        Args:
            simulated_csi: [N, 52] complex simulated CSI
            target_csi: [N, 52] complex target (real) CSI
            
        Returns:
            total_loss: Weighted combination of losses
            loss_dict: Individual loss components
        """
        # Amplitude Loss (L2)
        sim_amp = simulated_csi.abs()
        tgt_amp = target_csi.abs()
        loss_amp = torch.nn.functional.mse_loss(sim_amp, tgt_amp)
        
        # Phase Loss (Angular)
        sim_phase = torch.angle(simulated_csi)
        tgt_phase = torch.angle(target_csi)
        # Wrap phase difference to [-pi, pi]
        phase_diff = torch.atan2(
            torch.sin(sim_phase - tgt_phase),
            torch.cos(sim_phase - tgt_phase)
        )
        loss_phase = (phase_diff ** 2).mean()
        
        # Temporal Coherence (smoothness across time)
        if simulated_csi.shape[0] > 1:
            sim_diff = torch.diff(simulated_csi, dim=0)
            tgt_diff = torch.diff(target_csi, dim=0)
            loss_temporal = torch.nn.functional.mse_loss(sim_diff.abs(), tgt_diff.abs())
        else:
            loss_temporal = torch.tensor(0.0, device=self.device)
        
        # Weighted total
        total_loss = (
            self.config.weight_csi_amplitude * loss_amp +
            self.config.weight_csi_phase * loss_phase +
            self.config.weight_temporal_coherence * loss_temporal
        )
        
        return total_loss, {
            'amplitude': loss_amp.item(),
            'phase': loss_phase.item(),
            'temporal': loss_temporal.item(),
            'total': total_loss.item()
        }
    
    def simulate_csi(
        self, 
        num_frames: int = 30,
        source_pos: Tuple[int, int, int] = (5, 5, 5)
    ) -> torch.Tensor:
        """
        Generate simulated CSI using the current Digital Twin state.
        
        Returns:
            csi: [num_frames, 52] complex CSI
        """
        orch = self.orchestrator
        
        # Get grid shape from FDTD solver
        grid_shape = (
            orch.diff_fdtd.solver.nx,
            orch.diff_fdtd.solver.ny,
            orch.diff_fdtd.solver.nz
        )
        
        # Build epsilon grid from learnable materials ONCE (shared across frames)
        concrete_eps = orch.learnable_materials("concrete")[0]
        eps_grid = torch.ones(grid_shape, device=self.device, requires_grad=True) * concrete_eps
        
        # Single forward pass with longer simulation to get multiple "frames"
        orch.diff_fdtd.solver.reset_grid()
        
        steps_per_frame = 3
        total_steps = num_frames * steps_per_frame
        
        # Source signal
        source_vals = torch.sin(
            torch.linspace(0, 10.0 * 3.14159, total_steps, device=self.device)
        )
        
        # Forward through differentiable FDTD (single tape for entire simulation)
        fields = orch.diff_fdtd(total_steps, source_pos, source_vals, eps_grid)
        
        # Extract CSI from final Ez field
        ez_field = fields[2]  # [nx, ny, nz]
        
        # Sample 52 points along diagonal to get subcarrier-like distribution
        csi_samples = []
        for i in range(52):
            x = int(i * (grid_shape[0] - 1) / 51)
            y = int(i * (grid_shape[1] - 1) / 51)
            z = grid_shape[2] // 2
            csi_samples.append(ez_field[x, y, z])
        
        csi_frame = torch.stack(csi_samples)  # [52]
        
        # Apply noise twin to create multiple "frames" with variations
        all_csi = []
        for t in range(num_frames):
            # Add frame-specific noise variation
            if hasattr(orch, 'noise_twin') and orch.noise_twin is not None:
                csi_complex = csi_frame + 0j  # Convert to complex
                csi_noisy = orch.noise_twin(csi_complex.unsqueeze(0), {}).squeeze(0)
                all_csi.append(csi_noisy)
            else:
                all_csi.append(csi_frame)
        
        return torch.stack(all_csi)  # [num_frames, 52]
    
    def calibrate(
        self, 
        target_csi: torch.Tensor,
        checkpoint_dir: Optional[Path] = None
    ) -> Dict:
        """
        Main calibration loop.
        
        Args:
            target_csi: [N, 52] complex target CSI from real hardware
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            results: Calibration results and final parameters
        """
        orch = self.orchestrator
        
        # Setup optimizers for different component groups
        param_groups = [
            {'params': orch.learnable_materials.parameters(), 'lr': self.config.lr_materials},
        ]
        
        if hasattr(orch, 'noise_twin') and orch.noise_twin is not None:
            param_groups.append(
                {'params': orch.noise_twin.parameters(), 'lr': self.config.lr_noise}
            )
        
        optimizer = optim.Adam(param_groups)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs, eta_min=1e-6
        )
        
        print(f"\n[Calibrator] Starting Calibration Loop")
        print(f"   - Target CSI Shape: {target_csi.shape}")
        print(f"   - Epochs: {self.config.epochs}")
        
        best_loss = float('inf')
        
        for epoch in tqdm(range(self.config.epochs), desc="Calibrating"):
            optimizer.zero_grad()
            
            # Warmup learning rate
            if epoch < self.config.warmup_epochs:
                warmup_factor = (epoch + 1) / self.config.warmup_epochs
                for pg in optimizer.param_groups:
                    pg['lr'] = pg['lr'] * warmup_factor
            
            # Simulate CSI
            num_frames = min(target_csi.shape[0], 30)  # Limit for memory
            simulated_csi = self.simulate_csi(num_frames=num_frames)
            
            # NaN guard
            if torch.isnan(simulated_csi).any():
                print(f"  [WARN] NaN in simulation at epoch {epoch}. Skipping.")
                continue
            
            # Compute loss
            loss, loss_dict = self.compute_csi_loss(
                simulated_csi, 
                target_csi[:num_frames].to(self.device)
            )
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                orch.learnable_materials.parameters(), 
                self.config.grad_clip_norm
            )
            
            optimizer.step()
            scheduler.step()
            
            # Record history
            self.calibration_history['loss'].append(loss_dict['total'])
            self.calibration_history['loss_amplitude'].append(loss_dict['amplitude'])
            self.calibration_history['loss_phase'].append(loss_dict['phase'])
            
            # Log
            if epoch % self.config.eval_every == 0:
                concrete_eps = orch.learnable_materials("concrete")[0].item()
                print(f"\n  Epoch {epoch:3d} | Loss: {loss_dict['total']:.6f} | "
                      f"Amp: {loss_dict['amplitude']:.4f} | Phase: {loss_dict['phase']:.4f} | "
                      f"Eps_Concrete: {concrete_eps:.4f}")
            
            # Track best
            if loss_dict['total'] < best_loss:
                best_loss = loss_dict['total']
            
            # Checkpoint
            if checkpoint_dir and epoch % self.config.save_every == 0:
                self._save_checkpoint(checkpoint_dir, epoch)
        
        # Final results
        results = {
            'best_loss': best_loss,
            'final_epsilon_concrete': orch.learnable_materials("concrete")[0].item(),
            'history': self.calibration_history
        }
        
        if hasattr(orch, 'noise_twin') and orch.noise_twin is not None:
            results['noise_params'] = {
                'thermal_dbm': orch.noise_twin.thermal_noise_dbm.item(),
                'iq_amp_error': orch.noise_twin.iq_amp_error.item(),
            }
        
        print(f"\n[Calibrator] Calibration Complete!")
        print(f"   - Best Loss: {best_loss:.6f}")
        print(f"   - Final Epsilon (Concrete): {results['final_epsilon_concrete']:.4f}")
        
        return results
    
    def _save_checkpoint(self, checkpoint_dir: Path, epoch: int):
        """Save calibration checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        orch = self.orchestrator
        
        checkpoint = {
            'epoch': epoch,
            'materials_state_dict': orch.learnable_materials.state_dict(),
            'history': self.calibration_history,
        }
        
        if hasattr(orch, 'noise_twin') and orch.noise_twin is not None:
            checkpoint['noise_twin_state_dict'] = orch.noise_twin.state_dict()
        
        torch.save(checkpoint, checkpoint_dir / f"calibration_epoch_{epoch}.pt")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load calibration checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        
        orch = self.orchestrator
        orch.learnable_materials.load_state_dict(checkpoint['materials_state_dict'])
        
        if 'noise_twin_state_dict' in checkpoint and hasattr(orch, 'noise_twin'):
            orch.noise_twin.load_state_dict(checkpoint['noise_twin_state_dict'])
        
        self.calibration_history = checkpoint.get('history', {})
        
        print(f"[Calibrator] Loaded checkpoint from epoch {checkpoint['epoch']}")


def calibrate_from_hdf5(
    orchestrator,
    hdf5_path: str,
    csi_key: str = 'csi',
    config: CalibrationConfig = None
) -> Dict:
    """
    Convenience function to calibrate from HDF5 file.
    
    Args:
        orchestrator: SovereignOrchestrator instance
        hdf5_path: Path to HDF5 file with target CSI
        csi_key: Key for CSI dataset in HDF5
        config: Calibration configuration
        
    Returns:
        Calibration results
    """
    with h5py.File(hdf5_path, 'r') as f:
        target_csi = torch.from_numpy(f[csi_key][:]).to(orchestrator.device)
    
    calibrator = DigitalTwinCalibrator(orchestrator, config)
    return calibrator.calibrate(target_csi)
