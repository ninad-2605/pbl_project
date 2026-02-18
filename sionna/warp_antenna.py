
import warp as wp
import torch
import math
from src.physics.sionna_clones.warp_complex import (
    Complex64, ComplexVec2, complex_exp, complex_mul, complex_add, complex_conj, complex_scale
)

# Constants
PI = 3.141592653589793
TWO_PI = 2.0 * PI

@wp.struct
class AntennaElement:
    """
    Represents a single antenna element within an array.
    """
    position: wp.vec3   # Position relative to array centroid
    orientation: wp.quat # Rotation from Global to Element Local Frame
    polarization_slant: float # 0=V, 90=H, +/-45
    pattern_id: int     # 0=Isotropic, 1=Dipole, 2=38.901

@wp.kernel
def compute_steering_vector_kernel(
    antenna_elements: wp.array(dtype=AntennaElement),
    k_vecs: wp.array(dtype=wp.vec3), # [N_dirs, 3] Direction vectors (normalized)
    wavenumber: float,               # 2*pi/lambda
    steering_vectors: wp.array(dtype=Complex64, ndim=2) # [N_dirs, N_elements]
):
    """
    Computes the array response vector (steering vector) for a set of directions.
    a(theta) = exp(-j * k * dot(r, k_hat))
    """
    dir_idx, el_idx = wp.tid()
    
    k_hat = k_vecs[dir_idx]
    element = antenna_elements[el_idx]
    r = element.position
    
    phase_val = wavenumber * wp.dot(r, k_hat)
    
    re = wp.cos(phase_val)
    im = wp.sin(phase_val)
    
    steering_vectors[dir_idx, el_idx] = Complex64(re, im)

class WarpAntennaArray:
    """
    Manages Antenna Array geometry and computations on GPU.
    """
    def __init__(self, positions: torch.Tensor, orientations: torch.Tensor = None, slants: torch.Tensor = None, pattern_ids: torch.Tensor = None, device='cuda'):
        self.device = device
        self.num_elements = positions.shape[0]
        
        # Defaults
        if orientations is None:
            # Default identity quaternion (0, 0, 0, 1) -> (x, y, z, w)
            orientations = torch.zeros((self.num_elements, 4), dtype=torch.float32, device=device)
            orientations[:, 3] = 1.0 # w=1
            
        if slants is None:
            slants = torch.zeros(self.num_elements, dtype=torch.float32, device=device)
        if pattern_ids is None:
            pattern_ids = torch.zeros(self.num_elements, dtype=torch.int32, device=device) + 2 # Default 38.901
            
        # Pack into struct array
        self.elements = wp.zeros(self.num_elements, dtype=AntennaElement, device=self.device)
        
        wp_pos = wp.from_torch(positions.contiguous(), dtype=wp.vec3)
        wp_rot = wp.from_torch(orientations.contiguous(), dtype=wp.quat) 
        wp_slants = wp.from_torch(slants.contiguous(), dtype=wp.float32)
        wp_ids = wp.from_torch(pattern_ids.contiguous(), dtype=wp.int32)
        
        # Store for external access (e.g. SovereignRT MIMO)
        self.slants_torch = slants

        
        wp.launch(
            kernel=self.pack_elements_kernel,
            dim=self.num_elements,
            inputs=[wp_pos, wp_rot, wp_slants, wp_ids, self.elements],
            device=self.device
        )
        
    @staticmethod
    def create_upa(rows=4, cols=4, spacing=0.5, device='cuda'):
        """
        Create Uniform Planar Array (UPA) centered at origin in YZ plane (facing X).
        """
        # Grid
        y = torch.linspace(-(cols-1)/2, (cols-1)/2, cols, device=device) * spacing
        z = torch.linspace(-(rows-1)/2, (rows-1)/2, rows, device=device) * spacing
        
        grid_y, grid_z = torch.meshgrid(y, z, indexing='xy')
        grid_x = torch.zeros_like(grid_y)
        
        pos = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
        return WarpAntennaArray(pos, device=device)

    @wp.kernel
    def pack_elements_kernel(
        pos: wp.array(dtype=wp.vec3),
        rot: wp.array(dtype=wp.quat),
        slants: wp.array(dtype=wp.float32),
        ids: wp.array(dtype=wp.int32),
        out: wp.array(dtype=AntennaElement)
    ):
        i = wp.tid()
        el = out[i]
        el.position = pos[i]
        el.orientation = rot[i]
        el.polarization_slant = slants[i]
        el.pattern_id = ids[i]
        out[i] = el

    def compute_steering_vectors(self, k_vecs: torch.Tensor, wavelength: float) -> torch.Tensor:
        """
        Compute steering vectors for batch of directions.
        k_vecs: [N_dirs, 3] normalized direction vectors.
        Returns: [N_dirs, N_elements] complex tensor
        """
        N_dirs = k_vecs.shape[0]
        wavenumber = 2.0 * math.pi / wavelength
        
        wp_k = wp.from_torch(k_vecs.contiguous(), dtype=wp.vec3)
        wp_sv = wp.zeros((N_dirs, self.num_elements), dtype=Complex64, device=self.device)
        
        wp.launch(
            kernel=compute_steering_vector_kernel,
            dim=(N_dirs, self.num_elements),
            inputs=[self.elements, wp_k, wavenumber, wp_sv],
            device=self.device
        )
        
        real_part = wp.zeros((N_dirs, self.num_elements), dtype=wp.float32, device=self.device)
        imag_part = wp.zeros((N_dirs, self.num_elements), dtype=wp.float32, device=self.device)
        
        wp.launch(
            kernel=self.unpack_complex_matrix_kernel,
            dim=(N_dirs, self.num_elements),
            inputs=[wp_sv, real_part, imag_part],
            device=self.device
        )
        
        return torch.complex(wp.to_torch(real_part), wp.to_torch(imag_part))

    @wp.kernel
    def unpack_complex_matrix_kernel(
        c_in: wp.array(dtype=Complex64, ndim=2),
        re_out: wp.array(dtype=wp.float32, ndim=2),
        im_out: wp.array(dtype=wp.float32, ndim=2)
    ):
        i, j = wp.tid()
        c = c_in[i, j]
        re_out[i, j] = c.real
        im_out[i, j] = c.imag
