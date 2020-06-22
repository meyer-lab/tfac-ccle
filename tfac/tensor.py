"""
Tensor decomposition methods
"""
import numpy as np
import tensorly as tl
from tensorly.decomposition import partial_tucker
from tensorly.metrics.regression import variance as tl_var
from tensorly.tenalg import mode_dot
from tfac.Data_Mod import form_tensor
from tfac.tensor import partial_tucker_decomp, find_R2X_partialtucker
from tfac.dataHelpers import importLINCSprotein

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


def reorient_factors(factors):
    """ Reorient factors based on the sign of the mean so that only the last factor can have negative means. """
    for index in range(len(factors) - 1):
        meann = np.sign(np.mean(factors[index], axis=0))
        assert meann.size == factors[0].shape[1]

        factors[index] *= meann
        factors[index + 1] *= meann

    return factors


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
    return partial_tucker(tensor, mode_list, rank, tol=1.0e-12)


#### For R2X Plots ###########################################################################


def find_R2X_partialtucker(tucker_output, orig):
    """Compute R2X for the tucker decomposition."""
    return R2X(mode_dot(tucker_output[0], tucker_output[1][0], 2), orig)

def flip_factors(tucker_output, components, treatments_array):
    frame_list = []
    for i in range(len(treatments_array)):
        df = pd.DataFrame(tucker_output[0][i])
        frame_list.append(df)
    for component in range(components):
        column_list = []
        for i in range(len(treatments_array)):
            column_list.append(pd.DataFrame(frame_list[i].iloc[:, component]))
        df = pd.concat(column_list, axis=1)
        av = df.values.mean()
        if(av < 0 and tucker_output[1][0][:, component].mean() < 0):
            tucker_output[1][0][:, component] *= -1
            for j in range(len(treatments)):
                tucker_output[0][j][:, component] *= -1
    return tucker_output