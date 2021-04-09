"""
Tensor decomposition methods.
"""
import numpy as np
import tensorly as tl
from tensorly.decomposition import partial_tucker
from tensorly.metrics.regression import variance as tl_var
from tensorly.tenalg import mode_dot
from .Data_Mod import form_tensor


tl.set_backend("numpy")  # Set the backend


def R2X(reconstructed, original):
    """ Calculates R2X of two tensors. """
    return 1.0 - tl_var(reconstructed - original) / tl_var(original)


#### Decomposition Methods ###################################################################


def partial_tucker_decomp(tensor, mode_list, rank):
    """Perform Partial Tucker decomposition.
    -----------------------------------------------
    Input:
        tensor: 3D data tensor
        mode_list: which mode(s) to apply tucker decomposition to
        rank: rank of decomposition
    Returns
        output[0]: core tensor
        output[1]: list of factor matrices
    """
    pre_flip_result = partial_tucker(tensor, mode_list, rank, tol=1.0e-12)
    result = flip_factors(pre_flip_result)
    return result


#### For R2X Plots ###########################################################################


def find_R2X_partialtucker(tucker_output, orig):
    """Compute R2X for the tucker decomposition."""
    return R2X(mode_dot(tucker_output[0], tucker_output[1][0], 2), orig)


###### To Flip Factors #########################################################################


def flip_factors(tucker_output):
    """For partial tucker OHSU factorization, flips protein and treatment/time factors if both negative for important values"""
    for component in range(tucker_output[0].shape[2]):
        av = 0.0
        for i in range(tucker_output[0].shape[0]):
            av += np.mean(tucker_output[0][i][:, component] ** 5)

        if av < 0 and tucker_output[1][0][:, component].mean() < 0:
            tucker_output[1][0][:, component] *= -1
            for j in range(tucker_output[0].shape[0]):
                tucker_output[0][j][:, component] *= -1
    return tucker_output


def decomp_to_flipped_factors(components):
    """"Parital Tucker decomposition and flipping factors.
    Returns flipped factors, treatment list, and times."""
    tensor, treatment_list, times = form_tensor()
    result = partial_tucker_decomp(tensor, [2], components)
    return result, treatment_list, times
