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


###### Factorization #########################################################################


def tensor_factor(protein, geneExp, component):
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
    #Protein Factors
    proteinFactors = partial_tucker(protein, [2], component, tol=1.0e-12)
    for component in range(proteinFactors[0].shape[2]):
        av = 0.0
        for i in range(proteinFactors[0].shape[0]):
            av += np.mean(proteinFactors[0][i][:, component] ** 5)

        if av < 0 and proteinFactors[1][0][:, component].mean() < 0:
            proteinFactors[1][0][:, component] *= -1
            for j in range(proteinFactors[0].shape[0]):
                proteinFactors[0][j][:, component] *= -1
    #Gene Expression Factors
    geneFactors = partial_tucker(geneExp, [2], component, tol=1.0e-12)
    for component in range(geneFactors[0].shape[2]):
        av = 0.0
        for i in range(geneFactors[0].shape[0]):
            av += np.mean(geneFactors[0][i][:, component] ** 5)

        if av < 0 and geneFactors[1][0][:, component].mean() < 0:
            geneFactors[1][0][:, component] *= -1
            for j in range(geneFactors[0].shape[0]):
                geneFactors[0][j][:, component] *= -1
    return proteinFactors, geneFactors
     
