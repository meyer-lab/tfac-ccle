"""
Tensor decomposition methods
"""
import numpy as np
import tensorly as tl
from tensorly.decomposition import non_negative_parafac, tucker
from tensorly.metrics.regression import variance as tl_var

tl.set_backend("numpy")  # Set the backend


def z_score_values(A, cell_dim):
    """ Function that takes in the values tensor and z-scores it. """
    assert cell_dim < tl.ndim(A)
    convAxes = tuple([i for i in range(tl.ndim(A)) if i != cell_dim])
    convIDX = [None] * tl.ndim(A)
    convIDX[cell_dim] = slice(None)

    sigma = tl.tensor(np.std(tl.to_numpy(A), axis=convAxes))
    return A / sigma[tuple(convIDX)]


def R2X(reconstructed, original):
    """ Calculates R2X of two tensors. """
    return 1.0 - tl_var(reconstructed - original) / tl_var(original)


def perform_decomposition(tensor, r, weightFactor=2):
    """ Perform PARAFAC decomposition. """
    weights, factors = non_negative_parafac(tensor, r, tol=1.0e-10, n_iter_max=6000, normalize_factors=True)
    factors[weightFactor] *= weights[np.newaxis, :]  # Put weighting in designated factor
    return factors


def perform_tucker(tensor, rank_list):
    """ Perform Tucker decomposition. """
    # index 0 is for core tensor, index 1 is for factors; output is a list of core and factors
    return tucker(tensor, rank_list, , n_iter_max=2000, tol=1.0e-10)


def find_R2X_tucker(values, out):
    """Compute R2X for the tucker decomposition."""
    return R2X(tl.tucker_to_tensor(out), values)


def find_R2X(values, factors):
    """Compute R2X from parafac. Note that the inputs values and factors are in numpy."""
    return R2X(tl.kruskal_to_tensor((np.ones(factors[0].shape[1]), factors)), values)
