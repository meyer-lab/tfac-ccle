import numpy as np
import pandas as pd
from ..tensor import decomp_to_flipped_factors
from ..dataHelpers import ohsu_data
from ..pseudoinvnorm import find_factors


def test_check_size():
    """Imports raw data, implements tensor decomposition and reverse projection.
    Checks factor sizes.
    Returns decomp and projection sizes."""
    # Data Import
    comps = 5  # Use 5 components
    _, _, _, _, _, RNAseq, _ = ohsu_data()
    RNAseq.drop("ensembl_gene_id", inplace=True, axis=1)
    # Tensor Decomp & Reverse Projection
    result, treatment_list, times = decomp_to_flipped_factors(comps)
    P_gene, _, W_gene = find_factors(result, RNAseq, treatment_list, times)
    gene_recon = np.matmul(W_gene.T, P_gene)

    assert gene_recon.shape[0] == RNAseq.shape[0]
    assert gene_recon.shape[1] == 15
    assert len(result[1][0][0, :]) == comps
    assert len(result[1][0][:, 0]) == 295
    assert len(result[0][0][0]) == comps
    assert len(result[0]) == (len(treatment_list))
    assert len(result[0][0]) == len(times)
