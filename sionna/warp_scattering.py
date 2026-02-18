
import warp as wp
import numpy as np

@wp.func
def lambertian_pattern(cos_theta_s: wp.float32) -> wp.float32:
    """
    Standard Lambertian diffuse reflection pattern.
    f_s = cos(theta_s) / pi
    """
    return wp.max(0.0, cos_theta_s) / 3.14159265359

@wp.func
def log_factorial(n: wp.int32) -> wp.float32:
    """Natural log of n! for n >= 0"""
    if n <= 1: return 0.0
    # Stirling's approx for large n? Or simple loop for small n?
    # For GPU kernels, simple loop is okay if n is small (<20).
    # But alpha can be larger.
    # Use Ramanujan or just a loop.
    res = float(0.0)
    for i in range(2, n + 1):
        res += wp.log(float(i))
    return res

@wp.func
def binom(n: wp.int32, k: wp.int32) -> wp.float32:
    """Binomial coefficient (n choose k)"""
    if k < 0 or k > n: return 0.0
    # result = exp(ln(n!) - ln(k!) - ln(n-k)!)
    ln_n = log_factorial(n)
    ln_k = log_factorial(k)
    ln_nmk = log_factorial(n - k)
    return wp.exp(ln_n - ln_k - ln_nmk)

@wp.func
def compute_normalization_factor(
    cos_theta_i: wp.float32, 
    alpha_r: wp.int32, 
    alpha_i: wp.int32,
    lambda_param: wp.float32
) -> wp.float32:
    """
    Computes F_{alpha_r, alpha_i} from Degli-Esposti 2007.
    """
    sin_theta_i = wp.sqrt(wp.max(0.0, 1.0 - cos_theta_i * cos_theta_i))
    
    f_alpha_i = float(0.0)
    f_alpha_r = float(0.0)
    k_n = float(0.0)
    
    alpha_max = wp.max(alpha_r, alpha_i)
    
    PI = 3.14159265359
    
    for j in range(alpha_max + 1):
        # Even j
        i_j = float(0.0)
        if (j % 2) == 0:
            i_j = 2.0 * PI / float(j + 1)
        else:
            # Odd j
            n = (j - 1) // 2
            
            # Compute k_n term
            # v = (sin_theta_i)^(2n) * binom(2n, n) / 2^(2n)
            v = wp.pow(sin_theta_i, float(2 * n))
            v *= binom(2 * n, n)
            v /= wp.pow(2.0, float(2 * n))
            
            k_n += v
            
            i_j = cos_theta_i * k_n * 2.0 * PI / float(j + 1)
            
        # Update sums
        if j <= alpha_i:
            f_alpha_i += i_j * binom(alpha_i, j)
            
        if j <= alpha_r:
            f_alpha_r += i_j * binom(alpha_r, j)
            
    f_alpha_i /= wp.pow(2.0, float(alpha_i))
    f_alpha_r /= wp.pow(2.0, float(alpha_r))
    
    return lambda_param * f_alpha_r + (1.0 - lambda_param) * f_alpha_i

@wp.func
def backscattering_pattern(
    k_i: wp.vec3, # Incident vector (pointing TO surface)
    k_s: wp.vec3, # Scattered vector (pointing AWAY from surface)
    n: wp.vec3,   # Surface Normal
    alpha_r_float: wp.float32,   
    alpha_i_float: wp.float32,   
    lambda_param: wp.float32
) -> wp.float32:
    """
    Degli-Esposti directional scattering pattern with Rigorous Normalization.
    """
    # 1. Cosines with Normal
    # k_i points TO surface. Incidence angle theta_i is angle between -k_i and n.
    neg_k_i = -k_i 
    cos_theta_i = wp.dot(neg_k_i, n)
    cos_theta_s = wp.dot(k_s, n)
    
    # Clamp for safety
    cos_theta_i = wp.max(0.0, cos_theta_i)
    cos_theta_s = wp.max(0.0, cos_theta_s)
    
    # 2. Specular Direction (Reflection of k_i)
    # k_spec = k_i - 2*dot(k_i, n)*n (Standard reflection formula)
    # Check sign: k_i is incoming.
    # r = i - 2(i.n)n
    k_spec = k_i - 2.0 * wp.dot(k_i, n) * n
    
    # 3. Angle from Specular (Psi_r)
    # cos_psi_r = dot(k_s, k_spec)
    cos_psi_r = wp.dot(k_s, k_spec)
    
    # 4. Angle from Incident (Psi_i) -> Backscattering
    # Backscatter direction is -k_i.
    # cos_psi_i = dot(k_s, -k_i)
    # Note: Sionna uses angle between scattered and incident ray?
    # "Angle between the scattering direction and the incident direction" usually means dot(k_s, k_i).
    # But backscattering lobe peaks at -k_i.
    # Degli-Esposti Eq 8 involves ((1+cos(alpha))/2)^n.
    # If cos_psi_i corresponds to backscattering, it should peak when k_s = -k_i.
    # dot(k_s, -k_i) = 1.
    cos_psi_i = wp.dot(k_s, neg_k_i)
    
    # 5. Compute Pattern
    # S_r term (Specular Lobe)
    v_r = wp.pow((1.0 + cos_psi_r) * 0.5, alpha_r_float)
    
    # S_i term (Incident/Backscatter Lobe)
    v_i = wp.pow((1.0 + cos_psi_i) * 0.5, alpha_i_float)
    
    # Pattern W
    # w = lambda * v_r + (1-lambda) * v_i  <-- Is it additive?
    # Degli-Esposti Eq 8:
    # f_s = (1/pi) * [ (1-lambda)*cos_theta_s + 2*lambda*v_r ] * v_i
    # Wait, my previous code had:
    # diffuse = (1-lambda)*cos_theta_s
    # specular = 2*lambda*v_r
    # shape = (diffuse + specular) * v_i
    
    # This structure Matches the code I read earlier.
    # So `cos_psi_r` is used for the inner term. `cos_psi_i` modulates everything.
    
    diffuse = (1.0 - lambda_param) * cos_theta_s
    specular = 2.0 * lambda_param * v_r
    
    w_unnorm = wp.max(0.0, diffuse + specular) * v_i
    
    # 6. Normalization
    # We pass cos_theta_i and alphas to the helper.
    # Note: The helper assumes this specific shape structure.
    # Cast alphas to int for binomial
    alpha_r = wp.int32(alpha_r_float)
    alpha_i = wp.int32(alpha_i_float)
    
    f_norm = compute_normalization_factor(cos_theta_i, alpha_r, alpha_i, lambda_param)
    
    if f_norm < 1e-6: f_norm = 1.0
    
    return (w_unnorm / f_norm) / 3.14159265359

@wp.kernel
def compute_scattering_kernel(
    k_i: wp.array(dtype=wp.vec3),
    k_s: wp.array(dtype=wp.vec3),
    n: wp.array(dtype=wp.vec3),
    pattern_type: wp.int32, # 0 = Lambertian, 1 = Backscattering
    alpha_parameters: wp.array(dtype=wp.float32), # [alpha_r, alpha_i, lambda]
    output_fs: wp.array(dtype=wp.float32)
):
    tid = wp.tid()
    
    normal = n[tid]
    # Simple check for bad normals
    if wp.length(normal) < 0.1:
        output_fs[tid] = 0.0
        return
        
    ki_vec = k_i[tid]
    ks_vec = k_s[tid]
    
    # Normalize inputs just in case
    ki_vec = wp.normalize(ki_vec)
    ks_vec = wp.normalize(ks_vec)
    normal = wp.normalize(normal)
    
    if pattern_type == 0:
        # Lambertian only needs cos_theta_s
        cos_theta_s = wp.max(0.0, wp.dot(ks_vec, normal))
        output_fs[tid] = lambertian_pattern(cos_theta_s)
    else:
        output_fs[tid] = backscattering_pattern(
            ki_vec,
            ks_vec,
            normal,
            alpha_parameters[0],
            alpha_parameters[1],
            alpha_parameters[2]
        )
