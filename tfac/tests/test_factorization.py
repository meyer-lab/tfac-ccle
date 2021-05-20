import numpy as np
from ..tensor import tensor_factor
from ..dataHelpers import all_data_import, ohsu_data
from ..pseudoinvnorm import find_factors


def test_check_size():
    """Imports raw data, implements tensor decomposition and reverse projection.
    Checks factor sizes.
    Returns decomp and projection sizes."""
    # Data Import
    comps = 5  # Use 5 components
    protein, _ = all_data_import()
    _, _, _, _, _, RNAseq, _ = ohsu_data()
    RNAseq.drop("ensembl_gene_id", inplace=True, axis=1)
    treatment_list = ["BMP2", "EGF", "HGF", "IFNg", "OSM", "PBS", "TGFb"]
    times = ["0", "1", "4", "8", "24", "48"]
    # Tensor Decomp & Reverse Projection
    proteinFactors, _ = tensor_factor(protein, _, comps)
    P_gene, _, W_gene = find_factors(proteinFactors, RNAseq, treatment_list, times)
    gene_recon = np.matmul(W_gene.T, P_gene)

    assert gene_recon.shape[0] == RNAseq.shape[0]
    assert gene_recon.shape[1] == 15
    assert len(proteinFactors[1][0][0, :]) == comps
    assert len(proteinFactors[1][0][:, 0]) == 295
    assert len(proteinFactors[0][0][0]) == comps
    assert len(proteinFactors[0]) == (len(treatment_list))
    assert len(proteinFactors[0][0]) == len(times)
