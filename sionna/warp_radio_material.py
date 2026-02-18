
import warp as wp
import numpy as np
from src.physics.sionna_clones.warp_complex import (
    Complex64, complex_add, complex_sub, complex_mul, complex_div, complex_sqrt, complex_exp
)

@wp.struct
class MaterialData:
    """Struct for passing material properties to Warp kernels."""
    relative_permittivity: wp.float32
    conductivity: wp.float32
    thickness: wp.float32
    scattering_coefficient: wp.float32
    xpd_coefficient: wp.float32

class WarpRadioMaterial:
    """
    Enhanced RadioMaterial with MDL-grade logic and ITU-R P.2040 tables.
    """
    
    # ITU-R P.2040-3 Table 3: Material parameters
    # Key: (a, b, c, d) where eps_r = a*f^b, sigma = c*f^d (f in GHz)
    ITU_TABLE = {
        "concrete": (5.31, 0.0, 0.0326, 0.8095),
        "brick": (3.75, 0.0, 0.038, 0.0),
        "wood": (1.99, 0.0, 0.002, 1.0),
        "glass": (6.27, 0.0, 0.004, 1.19),
        "ceiling_board": (1.50, 0.0, 0.0005, 1.16),
        "chipboard": (2.58, 0.0, 0.0217, 0.78),
        "floorboard": (3.66, 0.0, 0.0044, 1.35),
        "metal": (1.0, 0.0, 1e7, 0.0), # Simplification for high conductivity
        "plasterboard": (2.94, 0.0, 0.0116, 0.7076),
        "dry_rock": (3.0, 0.0, 0.0001, 1.0),
    }

    def __init__(self, 
                 name: str, 
                 material_type: str = None, 
                 relative_permittivity: float = 1.0, 
                 conductivity: float = 0.0,
                 thickness: float = 0.1,
                 scattering_coefficient: float = 0.0,
                 xpd_coefficient: float = 0.0):
        
        self.name = name
        self.thickness = thickness
        self.scattering_coefficient = scattering_coefficient
        self.xpd_coefficient = xpd_coefficient
        
        if material_type and material_type.lower() in self.ITU_TABLE:
            self.a, self.b, self.c, self.d = self.ITU_TABLE[material_type.lower()]
            self.dynamic_itu = True
            # Default to 2.4GHz for initial values
            self.update_itu_properties(2.4e9)
        else:
            self.relative_permittivity = relative_permittivity
            self.conductivity = conductivity
            self.dynamic_itu = False

    def update_itu_properties(self, frequency_hz: float):
        """Update properties based on ITU table for a given frequency."""
        if not self.dynamic_itu:
            return
        
        f_ghz = frequency_hz / 1e9
        self.relative_permittivity = float(self.a * (f_ghz ** self.b))
        self.conductivity = float(self.c * (f_ghz ** self.d))

    def get_data_struct(self) -> MaterialData:
        """Returns a MaterialData struct for this material."""
        data = MaterialData()
        data.relative_permittivity = self.relative_permittivity
        data.conductivity = self.conductivity
        data.thickness = self.thickness
        data.scattering_coefficient = self.scattering_coefficient
        data.xpd_coefficient = self.xpd_coefficient
        return data

@wp.func
def get_complex_permittivity(mat: MaterialData, freq_hz: wp.float32) -> Complex64:
    """Exact complex permittivity calculation."""
    omega = 2.0 * 3.14159265359 * freq_hz
    eps_0 = 8.8541878128e-12
    
    imag = mat.conductivity / (omega * eps_0)
    return Complex64(mat.relative_permittivity, -imag)

@wp.func
def compute_fresnel_slab(cos_theta_i: wp.float32, 
                         eta: Complex64, 
                         thickness: wp.float32, 
                         freq_hz: wp.float32):
    """
    MDL-grade Fresnel Reflection/Transmission for a dielectric slab.
    Uses Fabry-Pérot interference for multi-bounce inside the slab.
    Returns (R_te, R_tm) as Complex64.
    """
    # 1. Complex refractive index n2 = sqrt(eta)
    n2 = complex_sqrt(eta)
    
    # 2. Snell's law: sin2 = sin1 / n2 (complex)
    sin_theta_i_sq = 1.0 - cos_theta_i * cos_theta_i
    sin_theta_i = wp.sqrt(wp.max(0.0, sin_theta_i_sq))
    sin_theta_t = complex_div(Complex64(sin_theta_i, 0.0), n2)
    
    # cos2 = sqrt(1 - sin2^2)
    sin_theta_t_sq = complex_mul(sin_theta_t, sin_theta_t)
    cos_theta_t = complex_sqrt(complex_sub(Complex64(1.0, 0.0), sin_theta_t_sq))
    
    # 3. Single-interface Fresnel coefficients
    n2_cos_t = complex_mul(n2, cos_theta_t)
    n1_cos_i = Complex64(cos_theta_i, 0.0)
    
    # r12_te = (n1*cos1 - n2*cos2) / (n1*cos1 + n2*cos2)
    r12_te = complex_div(complex_sub(n1_cos_i, n2_cos_t), complex_add(n1_cos_i, n2_cos_t))
    
    # r12_tm = (n2*cos1 - n1*cos2) / (n2*cos1 + n1*cos2)
    n2_cos_i = Complex64(n2.real * cos_theta_i, n2.imag * cos_theta_i)
    r12_tm = complex_div(complex_sub(n2_cos_i, cos_theta_t), complex_add(n2_cos_i, cos_theta_t))
    
    # 4. Slab multi-bounce (Fabry-Pérot) if thickness > 0
    if thickness > 0.001:
        # One-way phase: phi = k * d * n2 * cos(theta_t)
        # k = 2*pi*f/c
        c0 = 299792458.0
        k0 = 2.0 * 3.14159265359 * freq_hz / c0
        
        # one-way phase: phi = k * d * n2 * cos(theta_t)
        n2_cos_t_scaled = complex_mul(n2, cos_theta_t)
        
        # phase_val = j * k0 * thickness * n2 * cos_theta_t 
        phase_val = complex_mul(Complex64(0.0, k0 * thickness), n2_cos_t_scaled)
        exp_phi = complex_exp(phase_val)
        
        # r21 = -r12 (from the other side)
        r21_te = Complex64(-r12_te.real, -r12_te.imag)
        r21_tm = Complex64(-r12_tm.real, -r12_tm.imag)
        
        # Slab reflection: R_slab = r12 * (1 - exp(-2j*phi)) / (1 - r12^2 * exp(-2j*phi))
        r12_te_sq = complex_mul(r12_te, r12_te)
        r12_tm_sq = complex_mul(r12_tm, r12_tm)
        
        # exp_neg_2phi for denominator (round-trip)
        neg_2_phase = Complex64(-2.0 * phase_val.real, -2.0 * phase_val.imag)
        exp_neg_2phi = complex_exp(neg_2_phase)
        
        one = Complex64(1.0, 0.0)
        
        # Numerator R: r12 * (1 - exp_neg_2phi)
        num_te = complex_mul(r12_te, complex_sub(one, exp_neg_2phi))
        num_tm = complex_mul(r12_tm, complex_sub(one, exp_neg_2phi))
        
        # Denominator: 1 - r12^2 * exp_neg_2phi
        den_te = complex_sub(one, complex_mul(r12_te_sq, exp_neg_2phi))
        den_tm = complex_sub(one, complex_mul(r12_tm_sq, exp_neg_2phi))
        
        r_te = complex_div(num_te, den_te)
        r_tm = complex_div(num_tm, den_tm)
        
        # --- TRANSMISSION COEFFICIENTS (T) ---
        # T_slab = t12 * t21 * exp(-j*phi) / (1 - r12^2 * exp(-2j*phi))
        # For symmetric slab: t12*t21 = 1 - r12^2 ? (Fresnel identity t*t' = 1-r^2 depends on normalization)
        # Standard form: t12 = 1 + r12. t21 = 1 + r21 = 1 - r12.
        # t12*t21 = (1+r)*(1-r) = 1 - r^2.
        
        t12_t21_te = complex_sub(one, r12_te_sq)
        t12_t21_tm = complex_sub(one, r12_tm_sq)
        
        # Numerator T: (1-r^2) * exp(-j*phi)
        # exp(-j*phi) = exp(-phase_val)
        neg_phase = Complex64(-phase_val.real, -phase_val.imag)
        exp_neg_phi = complex_exp(neg_phase)
        
        num_t_te = complex_mul(t12_t21_te, exp_neg_phi)
        num_t_tm = complex_mul(t12_t21_tm, exp_neg_phi)
        
        # Denominator is same as R
        t_te = complex_div(num_t_te, den_te)
        t_tm = complex_div(num_t_tm, den_tm)
        
        return r_te, r_tm, t_te, t_tm
    
    # No slab: return single-interface (T = 1+R)
    one = Complex64(1.0, 0.0)
    t12_te = complex_add(one, r12_te) # approx
    t12_tm = complex_add(one, r12_tm) # approx
    
    return r12_te, r12_tm, t12_te, t12_tm
