"""
Tensor decomposition methods.
"""
import numpy as np
import tensorly as tl
from tensorly.decomposition import partial_tucker
from tensorly.metrics.regression import variance as tl_var
from tensorly.tenalg import mode_dot
from .dataHelpers import importLINCSprotein, ohsu_data
from .Data_Mod import form_tensor
from .pseudoinvnorm import find_factors


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
    pre_flip_result = partial_tucker_decomp(tensor, [2], components)
    result = flip_factors(pre_flip_result)
    return result, treatment_list, times

#### Unit Test ###################################################################


def check_size(comps):
    """Imports raw data, implements tensor decomposition and reverse projection. 
    Checks factor sizes. 
    Returns decomp and projection sizes."""
    #Data Import
    _, _, _, _, _, RNAseq, _ = ohsu_data()
    RNAseq.drop("ensembl_gene_id", inplace=True, axis=1)
    #Tensor Decomp & Reverse Projection
    result, treatment_list, times = decomp_to_flipped_factors(comps)
    P_gene, _, _, W_gene = find_factors(result, RNAseq, treatment_list, times)
    gene_recon = np.matmul(W_gene.T, P_gene)
    if(gene_recon.shape[0] == RNAseq.shape[0] and gene_recon.shape[1] == 15):
        print("Gene reverse projection size matches original data set (genes by treatment-time): " 
        + str(gene_recon.shape))
    else:
        print("Gene reverse projection size doesn't match original data set.")
    if(len(result[1][0][0, :]) == comps and len(result[1][0][:, 0]) == 295):
        print("Protein factor sizes match expected values: 295 proteins by " + str(comps) + " components.")
    else:
        print("Protein factor size doesn't match.")
    if(len(result[0][0][0]) == comps and len(result[0]) == (len(treatment_list)) and len(result[0][0]) == len(times)):
        print("Core tensor size matches expected sizes of 7 treatments, 6 times, " + str(comps) + " components.")
    else:
        print("Core tensor size doesn't match.")

