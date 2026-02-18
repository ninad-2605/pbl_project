"""
Tier 1: Material Dielectrics (Expanded).
Defines the electromagnetic properties of matter at 2.4/5 GHz.
Includes Biological Tissues and Environmental scaling.
"""

from dataclasses import dataclass
import torch
import math

@dataclass
class MaterialProp:
    name: str
    epsilon_r: float  # Relative Permittivity (Real)
    conductivity: float # Sigma (S/m)
    color: tuple # RGB for Debug Vis
    humidity_sensitivity: float = 0.0 # dEpsilon/dHumidity

class DielectricDatabase:
    """
    The Periodic Table of RF Elements.
    Values at 2.45 GHz.
    Sources: IT'IS Foundation, Gabriel dispersion models.
    """
    def __init__(self):
        self._db = {
            # --- Conductors ---
            'metal':    MaterialProp('metal', 1.0, 1.0e7, (0.8, 0.8, 0.9)),
            'copper':   MaterialProp('copper', 1.0, 5.8e7, (1.0, 0.5, 0.2)),

            # --- Insulators (Building) ---
            'air':      MaterialProp('air', 1.0, 0.0, (0.0, 0.0, 0.0)),
            'vacuum':   MaterialProp('vacuum', 1.0, 0.0, (0.0, 0.0, 0.0)),
            'concrete': MaterialProp('concrete', 5.31, 0.07, (0.5, 0.5, 0.5), humidity_sensitivity=0.1),
            'wood':     MaterialProp('wood', 1.99, 0.012, (0.6, 0.4, 0.2), humidity_sensitivity=0.2),
            'glass':    MaterialProp('glass', 6.27, 0.01, (0.8, 0.9, 1.0)),
            'drywall':  MaterialProp('drywall', 2.94, 0.016, (0.9, 0.9, 0.8)),
            'brick':    MaterialProp('brick', 3.75, 0.038, (0.6, 0.2, 0.2)),
            
            # --- Biological (Anatomical Sovereign) ---
            # Based on 2.45 GHz IT'IS Data
            'skin_dry': MaterialProp('skin_dry', 38.0, 1.46, (0.9, 0.7, 0.6)),
            'skin_wet': MaterialProp('skin_wet', 43.0, 1.80, (0.8, 0.6, 0.5)),
            'fat':      MaterialProp('fat', 5.28, 0.10, (0.9, 0.9, 0.6)), # Low attenuation
            'muscle':   MaterialProp('muscle', 52.7, 1.74, (0.9, 0.2, 0.2)), # High attenuation
            'bone':     MaterialProp('bone', 11.4, 0.39, (0.9, 0.9, 0.9)),
            'blood':    MaterialProp('blood', 58.3, 2.54, (0.8, 0.0, 0.0)),
            
            # Internal Organs (New)
            'heart':    MaterialProp('heart', 54.8, 2.26, (0.8, 0.1, 0.1)),
            'lung_inf': MaterialProp('lung_inflated', 20.5, 0.80, (0.9, 0.6, 0.6)), # Filled with air
            'lung_def': MaterialProp('lung_deflated', 48.4, 1.69, (0.8, 0.4, 0.4)), # Collapsed
            'liver':    MaterialProp('liver', 43.0, 1.69, (0.6, 0.3, 0.3)),
            'brain':    MaterialProp('brain', 42.5, 1.51, (0.9, 0.8, 0.7))
        }

    def get(self, name: str) -> MaterialProp:
        return self._db.get(name.lower(), self._db['concrete'])

    def get_complex_permittivity(self, name: str, freq_hz: float = 2.45e9, humidity: float = 0.0) -> complex:
        """
        Calculate complex permittivity: ε* = ε' - j(σ / ωε0)
        Includes Humidity scaling for hygroscopic materials.
        """
        mat = self.get(name)
        omega = 2 * math.pi * freq_hz
        epsilon_0 = 8.854e-12
        
        # Humidity Scaling (Linear approx)
        # Humidity 0.0 to 1.0 (0% to 100%)
        eps_prime = mat.epsilon_r + (mat.humidity_sensitivity * humidity * 10.0) 
        
        # Sigma also increases with humidity (water is conductive)
        sigma = mat.conductivity * (1.0 + humidity) if mat.humidity_sensitivity > 0 else mat.conductivity
        
        loss_tangent = sigma / (omega * epsilon_0)
        return complex(eps_prime, -loss_tangent * eps_prime)

class LearnableDielectricDatabase(torch.nn.Module):
    """
    Tier 4: The Digital Twin Material Engine.
    Wraps DielectricDatabase and makes material properties learnable.
    Optimization is performed via Gradient Descent against real CSI data.
    """
    def __init__(self, base_db: DielectricDatabase):
        super().__init__()
        self.params = torch.nn.ParameterDict()
        
        # Initialize from base_db
        # We store parameters in log-space to enforce positivity (Value = exp(Param))
        for name, prop in base_db._db.items():
            if name in ['air', 'vacuum']: continue # Keep constants constant
            
            # Epsilon_r (Relative Permittivity)
            # Normal Concrete: 5.31 -> log(5.31) = 1.67
            self.params[f"{name}_eps_log"] = torch.nn.Parameter(
                torch.tensor(math.log(prop.epsilon_r), dtype=torch.float32)
            )
            
            # Conductivity (Sigma)
            # Normal Concrete: 0.07 -> log(0.07) = -2.66
            # Use a small epsilon to avoid log(0)
            self.params[f"{name}_sigma_log"] = torch.nn.Parameter(
                torch.tensor(math.log(max(1e-6, prop.conductivity)), dtype=torch.float32)
            )

    def forward(self, name: str) -> torch.Tensor:
        """Returns [epsilon_r, sigma] for the given material."""
        name = name.lower()
        if name in ['air', 'vacuum']:
            return torch.tensor([1.0, 0.0], device=self.params['concrete_eps_log'].device)
            
        eps = torch.exp(self.params.get(f"{name}_eps_log", self.params['concrete_eps_log']))
        sigma = torch.exp(self.params.get(f"{name}_sigma_log", self.params['concrete_sigma_log']))
        
        return torch.stack([eps, sigma])

    def get_complex_permittivity(self, name: str, freq_hz: float = 2.45e9) -> torch.Tensor:
        """
        Calculate complex permittivity as a Differentiable Tensor.
        Returns: Complex64 Tensor
        """
        props = self.forward(name)
        eps_prime = props[0]
        sigma = props[1]
        
        omega = 2 * math.pi * freq_hz
        epsilon_0 = 8.854e-12
        
        loss_tangent = sigma / (omega * epsilon_0)
        
        # ε* = ε_prime * (1 - j*loss_tangent)
        real = eps_prime
        imag = -loss_tangent * eps_prime
        
        return torch.complex(real, imag)
