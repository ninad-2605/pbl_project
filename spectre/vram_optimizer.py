
import torch
try:
    import pynvml
except ImportError:
    pynvml = None
import time
from typing import Optional

class AdaptiveBatchSizer:
    """
    Tier 37 Redline Strategy: Adaptive VRAM Management.
    Dynamically scales batch sizes to target 90-95% VRAM utilization.
    """
    def __init__(self, target_utilization: float = 0.92, device_id: int = 0):
        self.target_utilization = target_utilization
        self.device_id = device_id
        
        try:
            if pynvml is None:
                raise ImportError("pynvml not found")
                
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            self.info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            self.total_vram = self.info.total
            self.has_nvml = True
        except Exception as e:
            # print(f"Warning: NVML initialization failed: {e}. Falling back to conservative sizing.")
            self.has_nvml = False
            self.total_vram = 0

    def get_vram_status(self):
        """Returns (used, total, utilization_fraction)"""
        if not self.has_nvml:
            return 0, 0, 0.0
        
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            used = info.used
            return used, self.total_vram, used / self.total_vram
        except:
            return 0, 0, 0.0

    def suggest_batch_size(self, current_batch_size: int, current_memory_usage: int) -> int:
        """
        Calculates optimal batch size based on linear scaling.
        NewSize = CurrentSize * (TargetVRAM / CurrentVRAM)
        """
        if not self.has_nvml or current_memory_usage <= 0:
            return current_batch_size
            
        used, total, util = self.get_vram_status()
        if total == 0: return current_batch_size
        
        target_vram = total * self.target_utilization
        
        # Simple linear projection: 
        # mem_per_sample = used / current_batch_size
        # But we need to account for 'baseline' memory (model weights, etc.)
        # baseline = used - (total_mem_at_current_batch)
        # For now, use a safer proportional scaling of the dynamic part:
        
        # If we are over target, aggressively downscale
        if util > self.target_utilization + 0.03:
            return int(current_batch_size * 0.8)
            
        # If we have headroom, slowly upscale
        if util < self.target_utilization - 0.05:
            # New batch size should fit in 'headroom'
            scaler = target_vram / max(1, used)
            new_size = int(current_batch_size * scaler)
            # Limit growth to 2x to avoid oscillation
            return min(new_size, current_batch_size * 2)
            
        return current_batch_size

    def __del__(self):
        if hasattr(self, 'has_nvml') and self.has_nvml and pynvml:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
