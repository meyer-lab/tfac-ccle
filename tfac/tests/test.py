import numpy as np
import pandas as pd
from ..tensor import decomp_to_flipped_factors
from ..Data_Mod import form_tensor
from ..dataHelpers import ohsu_data
from ..pseudoinvnorm import find_factors

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
    assert (gene_recon.shape[0] == RNAseq.shape[0] and gene_recon.shape[1] == 15), "Reverse projection factor sizes don't match."
    assert(len(result[1][0][0, :]) == comps and len(result[1][0][:, 0]) == 295), "Protein factor size doesn't match."
    assert(len(result[0][0][0]) == comps and len(result[0]) == (len(treatment_list)) and len(result[0][0]) == len(times)), "Core tensor size doesn't match."

    