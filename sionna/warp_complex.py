
import warp as wp

@wp.struct
class Complex64:
    real: wp.float32
    imag: wp.float32

@wp.func
def complex_add(a: Complex64, b: Complex64):
    return Complex64(a.real + b.real, a.imag + b.imag)

@wp.func
def complex_sub(a: Complex64, b: Complex64):
    return Complex64(a.real - b.real, a.imag - b.imag)

@wp.func
def complex_mul(a: Complex64, b: Complex64):
    return Complex64(a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real)

@wp.func
def complex_div(a: Complex64, b: Complex64):
    denom = b.real * b.real + b.imag * b.imag + 1e-12
    return Complex64((a.real * b.real + a.imag * b.imag) / denom, 
                     (a.imag * b.real - a.real * b.imag) / denom)

@wp.func
def complex_sqrt(z: Complex64):
    r = wp.sqrt(z.real * z.real + z.imag * z.imag)
    # sqrt(z) = sqrt((r+x)/2) + i * sign(y) * sqrt((r-x)/2)
    re = wp.sqrt(0.5 * (r + z.real))
    im = wp.sqrt(0.5 * (r - z.real))
    if z.imag < 0.0:
        im = -im
    return Complex64(re, im)

@wp.func
def complex_norm_sq(a: Complex64):
    return a.real * a.real + a.imag * a.imag

@wp.func
def complex_abs(a: Complex64) -> wp.float32:
    return wp.sqrt(a.real * a.real + a.imag * a.imag)

@wp.func
def complex_exp(z: Complex64) -> Complex64:
    """exp(a+bj) = exp(a) * (cos(b) + j*sin(b))"""
    ea = wp.exp(z.real)
    return Complex64(ea * wp.cos(z.imag), ea * wp.sin(z.imag))

@wp.func
def complex_scale(s: wp.float32, a: Complex64) -> Complex64:
    return Complex64(s * a.real, s * a.imag)

@wp.func
def complex_conj(a: Complex64) -> Complex64:
    return Complex64(a.real, -a.imag)

@wp.func
def complex_mul_real(a: Complex64, s: wp.float32) -> Complex64:
    return Complex64(a.real * s, a.imag * s)

@wp.struct
class ComplexMatrix22:
    """Jones Matrix representation (2x2 complex)"""
    # [m00 m01]
    # [m10 m11]
    m00: Complex64
    m01: Complex64
    m10: Complex64
    m11: Complex64

@wp.func
def mat22_mul(A: ComplexMatrix22, B: ComplexMatrix22):
    return ComplexMatrix22(
        complex_add(complex_mul(A.m00, B.m00), complex_mul(A.m01, B.m10)),
        complex_add(complex_mul(A.m00, B.m01), complex_mul(A.m01, B.m11)),
        complex_add(complex_mul(A.m10, B.m00), complex_mul(A.m11, B.m10)),
        complex_add(complex_mul(A.m10, B.m01), complex_mul(A.m11, B.m11))
    )

@wp.struct
class ComplexVec2:
    x: Complex64
    y: Complex64

@wp.func
def mat22_vec_mul(A: ComplexMatrix22, v: ComplexVec2) -> ComplexVec2:
    """
    Multiply 2x2 Complex Matrix by 2D Complex Vector.
    """
    # x' = m00*x + m01*y
    # y' = m10*x + m11*y
    
    t0 = complex_add(complex_mul(A.m00, v.x), complex_mul(A.m01, v.y))
    t1 = complex_add(complex_mul(A.m10, v.x), complex_mul(A.m11, v.y))
    return ComplexVec2(t0, t1)

@wp.func
def mat22_det(A: ComplexMatrix22) -> Complex64:
    # det = m00*m11 - m01*m10
    return complex_sub(complex_mul(A.m00, A.m11), complex_mul(A.m01, A.m10))

@wp.func
def mat22_inv(A: ComplexMatrix22) -> ComplexMatrix22:
    d = mat22_det(A)
    # Check singularity?
    # inv = (1/det) * [m11, -m01; -m10, m00]
    
    # 1/det
    one = Complex64(1.0, 0.0)
    inv_d = complex_div(one, d)
    
    # Adjunct
    m00 = complex_mul(inv_d, A.m11)
    m01 = complex_mul(inv_d, complex_sub(Complex64(0.0, 0.0), A.m01)) # -m01
    m10 = complex_mul(inv_d, complex_sub(Complex64(0.0, 0.0), A.m10)) # -m10
    m11 = complex_mul(inv_d, A.m00)
    
    return ComplexMatrix22(m00, m01, m10, m11)

@wp.func
def from_polar(r: wp.float32, theta: wp.float32) -> Complex64:
    return Complex64(r * wp.cos(theta), r * wp.sin(theta))
