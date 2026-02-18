
import warp as wp
import torch

@wp.kernel
def fermat_refine_reflection_kernel(
    tx_pos: wp.vec3,
    rx_pos: wp.vec3,
    path_points: wp.array(dtype=wp.vec3, ndim=2), # [path_idx, point_idx]
    box_normals: wp.array(dtype=wp.vec3, ndim=2), # Normals of the surfaces at each point
    learning_rate: wp.float32,
    iterations: wp.int32
):
    path_idx = wp.tid()
    num_bounces = path_points.shape[1]
    
    for _ in range(iterations):
        for i in range(num_bounces):
            p_curr = path_points[path_idx, i]
            n_curr = box_normals[path_idx, i]
            
            # Distance from previous point to current
            p_prev = tx_pos if i == 0 else path_points[path_idx, i-1]
            # Distance from current point to next
            p_next = rx_pos if i == num_bounces - 1 else path_points[path_idx, i+1]
            
            # Gradient: d/dP |P_prev - P| + d/dP |P - P_next|
            v_in = wp.normalize(p_curr - p_prev)
            v_out = wp.normalize(p_curr - p_next)
            
            grad = v_in + v_out
            
            # Project onto surface plane
            grad_proj = grad - n_curr * wp.dot(grad, n_curr)
            
            # Descent
            path_points[path_idx, i] = p_curr - grad_proj * learning_rate

class WarpPathOptimizer:
    """
    State-of-the-art Path Optimizer using Fermat's Principle.
    Refines path candidates found by SBR/Shooting.
    """
    def __init__(self, learning_rate: float = 0.1, iterations: int = 20):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def refine_reflections(self, tx_pos, rx_pos, path_points, surface_normals):
        """
        Refines a set of multi-bounce reflection points.
        tx_pos, rx_pos: wp.vec3
        path_points: wp.array(dtype=wp.vec3, ndim=2) shape (N, K)
        surface_normals: wp.array(dtype=wp.vec3, ndim=2) shape (N, K)
        """
        wp.launch(
            fermat_refine_reflection_kernel,
            dim=path_points.shape[0],
            inputs=[tx_pos, rx_pos, path_points, surface_normals, self.learning_rate, self.iterations]
        )
