
import warp as wp
import torch
import numpy as np

from src.physics.sionna_clones.warp_scene_object import WarpSceneObject
from src.physics.sionna_clones.warp_radio_material import WarpRadioMaterial, MaterialData, get_complex_permittivity, compute_fresnel_slab
from src.physics.sionna_clones.warp_complex import (
    Complex64, ComplexMatrix22, complex_mul, complex_add, complex_sub,
    complex_div, complex_exp, complex_abs, complex_scale, complex_sqrt
)
from src.physics.sionna_clones.warp_path_optimizer import WarpPathOptimizer
from src.physics.sionna_clones.warp_utd import fresnel, compute_utd_diffraction_coeff
from src.physics.sionna_clones.warp_scattering import lambertian_pattern, backscattering_pattern

# Path Types for Physics Dispatch
PATH_TYPE_NONE = wp.constant(0)
PATH_TYPE_REFLECTION = wp.constant(1)
PATH_TYPE_DIFFRACTION = wp.constant(2)
PATH_TYPE_SCATTERING = wp.constant(3)

@wp.func
def compute_antenna_gain_38901_kernel_func(
    theta: wp.float32, 
    phi: wp.float32
) -> wp.float32:
    # TR 38.901 Parameters (Standard 65 deg beamwidth)
    theta_3db = 65.0 / 180.0 * 3.1415926535
    phi_3db = 65.0 / 180.0 * 3.1415926535
    a_max = 30.0   # Max attenuation [dB]
    sla_v = 30.0   # Side-lobe attenuation [dB]
    g_e_max = 8.0  # Max element gain [dBi]
    
    # Vertical Pattern
    v_factor = (theta - 3.1415926535/2.0) / theta_3db
    a_v = -wp.min(12.0 * v_factor * v_factor, sla_v)
    
    # Horizontal Pattern
    h_factor = phi / phi_3db
    a_h = -wp.min(12.0 * h_factor * h_factor, a_max)
    
    # Combined Pattern
    a_db = -wp.min(-(a_v + a_h), a_max) + g_e_max
    
    # Linear gain (Power)
    a_db_clamped = wp.clamp(a_db, -100.0, 50.0)
    return wp.pow(10.0, a_db_clamped / 10.0)


class WarpSovereign:
    """
    The "No Compromise" Sovereign Engine.
    Fully native Warp implementation of Sionna RT logic with high-fidelity upgrades.
    """
    def __init__(self, frequency_hz=2.412e9, num_subcarriers=64, device="cuda"):
        self.frequency_hz = float(frequency_hz)
        self.wavelength = 299792458.0 / self.frequency_hz
        self.num_subcarriers = num_subcarriers
        self.device = device
        
        self.path_optimizer = WarpPathOptimizer(iterations=30)
        
        # Materials dictionary
        self.materials = {}
        self._init_default_materials()

    def _init_default_materials(self):
        for mat_name in ["concrete", "brick", "wood", "glass", "metal", "plasterboard"]:
            self.materials[mat_name] = WarpRadioMaterial(mat_name, material_type=mat_name)
            self.materials[mat_name].update_itu_properties(self.frequency_hz)

    def synthesize_csi(self, 
                       tx_pos,
                       rx_pos,
                       path_points,
                       path_normals,
                       path_materials,
                       path_types,
                       freq_hz=None):
        """
        Synthesize CSI using MDL Fresnel and Fermat-refined paths.
        
        Args:
            tx_pos: wp.vec3 - transmitter position
            rx_pos: wp.vec3 - receiver position
            path_points: wp.array(dtype=wp.vec3, ndim=2) [num_paths, max_bounces]
            path_normals: wp.array(dtype=wp.vec3, ndim=2) [num_paths, max_bounces]
            path_materials: wp.array(dtype=MaterialData, ndim=2) [num_paths, max_bounces]
            path_types: wp.array(dtype=wp.int32) [num_paths]
            
        Returns:
            csi_re, csi_im: wp.array(dtype=wp.float32, ndim=2) [num_paths, num_subcarriers]
        """
        if freq_hz is None:
            freq_hz = self.frequency_hz
            
        num_paths = path_points.shape[0]
        
        # Output arrays: real and imaginary parts of CSI per path per subcarrier
        csi_re = wp.zeros((num_paths, self.num_subcarriers), dtype=wp.float32)
        csi_im = wp.zeros((num_paths, self.num_subcarriers), dtype=wp.float32)
        
        # Subcarrier frequencies
        bw = 20.0e6  # 20MHz bandwidth (WiFi)
        f_center = freq_hz
        f_start = f_center - bw / 2.0
        f_step = bw / float(self.num_subcarriers)
        
        wp.launch(
            sovereign_physics_kernel,
            dim=num_paths,
            inputs=[
                tx_pos, rx_pos,
                path_points, path_normals, path_materials, path_types,
                wp.float32(f_center), wp.float32(f_start), wp.float32(f_step),
                self.num_subcarriers,
                csi_re, csi_im
            ]
        )
        return csi_re, csi_im


@wp.kernel
def sovereign_physics_kernel(
    tx_pos: wp.vec3,
    rx_pos: wp.vec3,
    path_points: wp.array(dtype=wp.vec3, ndim=2),
    path_normals: wp.array(dtype=wp.vec3, ndim=2),
    path_materials: wp.array(dtype=MaterialData, ndim=2),
    path_types: wp.array(dtype=wp.int32),
    freq_center: wp.float32,
    freq_start: wp.float32,
    freq_step: wp.float32,
    num_subcarriers: wp.int32,
    csi_re: wp.array(dtype=wp.float32, ndim=2),
    csi_im: wp.array(dtype=wp.float32, ndim=2)
):
    tid = wp.tid()
    PI = 3.14159265358979323846
    c0 = 299792458.0
    
    # --- 1. Path Geometry ---
    # Single-bounce path: Tx -> P1 -> Rx
    p1 = path_points[tid, 0]
    
    d_tx_p1 = wp.length(p1 - tx_pos)
    d_p1_rx = wp.length(rx_pos - p1)
    total_dist = d_tx_p1 + d_p1_rx
    
    # Guard against degenerate paths
    if total_dist < 1.0e-6:
        return
    
    # --- 2. MDL-Grade Fresnel Reflection ---
    dir_i = wp.normalize(p1 - tx_pos)
    norm = path_normals[tid, 0]
    cos_theta_i = wp.abs(wp.dot(dir_i, norm))
    
    mat = path_materials[tid, 0]
    eta = get_complex_permittivity(mat, freq_center)
    r_te, r_tm = compute_fresnel_slab(cos_theta_i, eta, mat.thickness, freq_center)
    
    # Average TE/TM for unpolarized (scalar) approximation
    # |R| = sqrt((|R_te|^2 + |R_tm|^2) / 2)
    r_te_mag_sq = r_te.real * r_te.real + r_te.imag * r_te.imag
    r_tm_mag_sq = r_tm.real * r_tm.real + r_tm.imag * r_tm.imag
    refl_mag = wp.sqrt((r_te_mag_sq + r_tm_mag_sq) * 0.5)
    
    # Reflection phase (use TE as reference)
    refl_phase = wp.atan2(r_te.imag, r_te.real)
    
    # --- 3. Friis Free-Space Path Loss (Corrected for Reflection vs Scattering) ---
    p_type = path_types[tid]
    wavelength = c0 / freq_center
    amp_total = float(0.0)
    
    if p_type == PATH_TYPE_REFLECTION:
        # SPECULAR REFLECTION: 1 / (d1 + d2)
        # amp = lambda / (4 * pi * total_dist)
        amp_total = (wavelength / (4.0 * PI * total_dist)) * refl_mag
    elif p_type == PATH_TYPE_SCATTERING:
        # BI-STATIC SCATTERING: 1 / (d1 * d2)
        # Radar Range Equation approx: amp = sqrt(rcs) * lambda / ((4*pi)^1.5 * d1 * d2)
        rcs = 0.5 # Default RCS for human body at 2.4GHz
        amp_const = (wavelength / (4.0 * PI)) * wp.sqrt(rcs / (4.0 * PI))
        amp_total = amp_const / (d_tx_p1 * d_p1_rx + 1e-6)
    else:
        # Default/LOS or unknown
        amp_total = (wavelength / (4.0 * PI * total_dist))
    
    # --- 3.5 TIER 21: ANTENNA GAIN (3GPP) ---
    # Simplified gain application: assume Z-up, zenith and azimuth
    # tx_gain = compute_antenna_gain_38901_kernel_func(theta_tx, phi_tx)
    # For now, apply g_max as a scalar until full pattern logic is wired
    g_max_linear = 6.3 # 8dBi element gain
    amp_total *= wp.sqrt(g_max_linear)
    
    # --- 4. Coherent CSI for Each Subcarrier ---
    for sc in range(num_subcarriers):
        f_sc = freq_start + wp.float32(sc) * freq_step
        lambda_sc = c0 / f_sc
        k_sc = 2.0 * PI / lambda_sc
        
        # Phase = k * total_distance + reflection_phase
        phase = k_sc * total_dist + refl_phase
        
        # CSI = amplitude * exp(-j * phase)
        csi_re[tid, sc] = amp_total * wp.cos(phase)
        csi_im[tid, sc] = amp_total * (-wp.sin(phase))
