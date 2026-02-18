"""
==============================================================================
Async Prefetch Pipeline - Zero-Idle Data Generation
==============================================================================

CPU prefetches iteration N+1 while GPU processes iteration N.
Uses producer-consumer pattern with pinned memory buffers.

Performance Target: Eliminate all CPU <-> GPU idle time.
"""

import torch
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass


@dataclass
class PrefetchConfig:
    """Configuration for async prefetch pipeline."""
    prefetch_depth: int = 5       # Number of items to keep ready
    num_workers: int = 2          # CPU threads for prefetching
    timeout_seconds: float = 10.0 # Max wait time for queue
    enable_pinned_memory: bool = True


class AsyncPrefetchPipeline:
    """
    CPU prefetches iteration N+1 while GPU processes N.
    Zero GPU idle time, Zero CPU idle time.
    
    Usage:
        pipeline = AsyncPrefetchPipeline(data_source, config)
        pipeline.start_prefetching()
        
        for _ in range(num_samples):
            gpu_data = pipeline.get_next_gpu()
            result = process_on_gpu(gpu_data)
            
        pipeline.stop()
    """
    
    def __init__(self, 
                 data_source,
                 config: Optional[PrefetchConfig] = None,
                 transform_fn: Optional[Callable] = None):
        """
        Args:
            data_source: Object with __len__ and get_frame(idx) methods
            config: PrefetchConfig instance
            transform_fn: Optional function to transform data before queueing
        """
        self.data_source = data_source
        self.config = config or PrefetchConfig()
        self.transform_fn = transform_fn
        
        self.prefetch_queue = queue.Queue(maxsize=self.config.prefetch_depth)
        self.executor = ThreadPoolExecutor(max_workers=self.config.num_workers)
        self.stop_flag = threading.Event()
        self.current_idx = 0
        self.total_items = len(data_source) if hasattr(data_source, '__len__') else float('inf')
        
        # Pre-allocated pinned buffers for fast CPU→GPU transfer
        self._buffers = None
        if self.config.enable_pinned_memory and torch.cuda.is_available():
            self._init_pinned_buffers()
            
        # Statistics
        self.stats = {
            'items_prefetched': 0,
            'queue_full_waits': 0,
            'buffer_reuses': 0,
        }
        
        self._prefetch_thread = None
        
    def _init_pinned_buffers(self):
        """Pre-allocate pinned memory buffers for reuse."""
        # SMPL-X typical sizes
        self._buffers = [{
            'vertices': torch.empty((10475, 3), dtype=torch.float32, pin_memory=True),
            'joints': torch.empty((127, 3), dtype=torch.float32, pin_memory=True),
            'pose': torch.empty((63,), dtype=torch.float32, pin_memory=True),
            'trans': torch.empty((3,), dtype=torch.float32, pin_memory=True),
            'betas': torch.empty((16,), dtype=torch.float32, pin_memory=True),
        } for _ in range(self.config.prefetch_depth)]
        
    def _get_buffer(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a reusable pinned buffer (round-robin)."""
        if self._buffers is None:
            return None
        self.stats['buffer_reuses'] += 1
        return self._buffers[idx % self.config.prefetch_depth]
        
    def _prefetch_worker(self, idx: int) -> Optional[Dict[str, Any]]:
        """CPU worker: Load data into pinned buffer."""
        if idx >= self.total_items:
            return None
            
        try:
            # Load from source (CPU-bound operation)
            raw_data = self.data_source.get_frame(idx)
            
            # Get reusable pinned buffer
            buf = self._get_buffer(idx)
            
            if buf is not None:
                # Copy into pinned buffer (fast memcpy, avoid allocation)
                for key in buf.keys():
                    if key in raw_data and raw_data[key] is not None:
                        src = raw_data[key]
                        if isinstance(src, torch.Tensor):
                            buf[key][:src.shape[0]].copy_(src)
                        else:
                            buf[key][:len(src)].copy_(torch.tensor(src))
                result = {'buffer': buf, 'frame_idx': idx, 'pinned': True}
            else:
                # Fallback: no pinned memory
                result = {'data': raw_data, 'frame_idx': idx, 'pinned': False}
            
            # Optional transform
            if self.transform_fn:
                result = self.transform_fn(result)
                
            return result
            
        except Exception as e:
            print(f"[AsyncPrefetch] Error loading frame {idx}: {e}")
            return None
    
    def start_prefetching(self, start_idx: int = 0):
        """Begin background prefetch from start_idx."""
        self.current_idx = start_idx
        self.stop_flag.clear()
        
        def prefetch_loop():
            while not self.stop_flag.is_set():
                # Check if queue has room
                if self.prefetch_queue.full():
                    self.stats['queue_full_waits'] += 1
                    threading.Event().wait(0.001)  # 1ms sleep
                    continue
                    
                if self.current_idx >= self.total_items:
                    break
                    
                # Submit prefetch task to thread pool
                future = self.executor.submit(
                    self._prefetch_worker, 
                    self.current_idx
                )
                result = future.result()
                
                if result is not None:
                    self.prefetch_queue.put(result)
                    self.stats['items_prefetched'] += 1
                    
                self.current_idx += 1
                    
        self._prefetch_thread = threading.Thread(target=prefetch_loop, daemon=True)
        self._prefetch_thread.start()
        
    def get_next_gpu(self, device: str = 'cuda') -> Dict[str, torch.Tensor]:
        """
        GPU consumer: Get next prefetched item.
        Transfers pinned buffer → GPU with non_blocking=True.
        
        Returns dict with tensors on GPU device.
        """
        try:
            item = self.prefetch_queue.get(timeout=self.config.timeout_seconds)
        except queue.Empty:
            raise RuntimeError("[AsyncPrefetch] Queue empty - prefetch may have stopped")
        
        if item['pinned']:
            buf = item['buffer']
            gpu_data = {
                key: tensor.to(device, non_blocking=True) 
                for key, tensor in buf.items()
            }
        else:
            # Non-pinned fallback
            gpu_data = {
                key: (val.to(device) if isinstance(val, torch.Tensor) else val)
                for key, val in item['data'].items()
            }
            
        gpu_data['frame_idx'] = item['frame_idx']
        return gpu_data
    
    def get_next_cpu(self) -> Dict[str, Any]:
        """Get next prefetched item without GPU transfer."""
        item = self.prefetch_queue.get(timeout=self.config.timeout_seconds)
        if item['pinned']:
            return item['buffer']
        return item['data']
        
    def stop(self):
        """Stop prefetching and cleanup."""
        self.stop_flag.set()
        if self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=2.0)
        self.executor.shutdown(wait=False)
        
    def get_stats(self) -> Dict[str, int]:
        """Get prefetch statistics."""
        return {
            **self.stats,
            'queue_size': self.prefetch_queue.qsize(),
            'current_idx': self.current_idx,
        }
        
    def __enter__(self):
        self.start_prefetching()
        return self
        
    def __exit__(self, *args):
        self.stop()


class TieredMemoryCache:
    """
    Multi-tier caching: VRAM → RAM → SSD
    LRU eviction policy with async promotion/demotion.
    """
    
    def __init__(self, 
                 vram_limit_mb: int = 4096,
                 ram_limit_mb: int = 16384,
                 ssd_cache_dir: Optional[str] = None):
        from collections import OrderedDict
        
        self.vram_limit = vram_limit_mb * 1024 * 1024
        self.ram_limit = ram_limit_mb * 1024 * 1024
        self.ssd_cache_dir = ssd_cache_dir
        
        self.vram_cache = OrderedDict()  # LRU ordered
        self.ram_cache = OrderedDict()
        self.ssd_index = {}  # key -> file path
        
        self.lock = threading.Lock()
        
        self._vram_used = 0
        self._ram_used = 0
        
    def _tensor_size(self, tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()
        
    def put(self, key: str, tensor: torch.Tensor, tier: str = 'vram'):
        """Store tensor in specified tier."""
        with self.lock:
            size = self._tensor_size(tensor)
            
            if tier == 'vram':
                while self._vram_used + size > self.vram_limit:
                    self._evict_vram_to_ram()
                self.vram_cache[key] = tensor.cuda()
                self._vram_used += size
                
            elif tier == 'ram':
                while self._ram_used + size > self.ram_limit:
                    self._evict_ram_to_ssd()
                self.ram_cache[key] = tensor.cpu().pin_memory()
                self._ram_used += size
                
    def get(self, key: str, device: str = 'cuda') -> Optional[torch.Tensor]:
        """Retrieve tensor, promoting through tiers as needed."""
        with self.lock:
            # VRAM hit
            if key in self.vram_cache:
                self.vram_cache.move_to_end(key)
                return self.vram_cache[key]
                
            # RAM hit - promote to VRAM
            if key in self.ram_cache:
                tensor = self.ram_cache.pop(key)
                size = self._tensor_size(tensor)
                self._ram_used -= size
                
                gpu_tensor = tensor.to(device, non_blocking=True)
                self.put(key, gpu_tensor, tier='vram')
                return gpu_tensor
                
            # SSD hit - load and promote
            if key in self.ssd_index:
                tensor = torch.load(self.ssd_index[key])
                self.put(key, tensor, tier='ram')
                return self.get(key, device)  # Recursive promotion
                
        return None
        
    def _evict_vram_to_ram(self):
        """Evict LRU item from VRAM to RAM."""
        if not self.vram_cache:
            return
        key, tensor = self.vram_cache.popitem(last=False)
        size = self._tensor_size(tensor)
        self._vram_used -= size
        self.put(key, tensor.cpu(), tier='ram')
        
    def _evict_ram_to_ssd(self):
        """Evict LRU item from RAM to SSD."""
        if not self.ram_cache or not self.ssd_cache_dir:
            return
        key, tensor = self.ram_cache.popitem(last=False)
        size = self._tensor_size(tensor)
        self._ram_used -= size
        
        import os
        path = os.path.join(self.ssd_cache_dir, f"{key}.pt")
        torch.save(tensor, path)
        self.ssd_index[key] = path


# Convenience function for data generation
def create_async_generator(data_source, num_samples: int, 
                           prefetch_depth: int = 5) -> AsyncPrefetchPipeline:
    """
    Create an async generator for data generation.
    
    Example:
        with create_async_generator(amass_loader, 1000) as pipeline:
            for _ in range(1000):
                gpu_data = pipeline.get_next_gpu()
                csi = ray_trace(gpu_data)
    """
    config = PrefetchConfig(
        prefetch_depth=prefetch_depth,
        num_workers=2,
        enable_pinned_memory=torch.cuda.is_available()
    )
    return AsyncPrefetchPipeline(data_source, config)
