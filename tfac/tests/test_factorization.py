import numpy as np
import tensorly as tl
from ..tensor import decomp_to_flipped_factors
from ..CMTF import perform_CMTF
from ..dataHelpers import ohsu_data
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
    assert len(result[1][0][0, :]) == comps
    assert len(result[1][0][:, 0]) == 295
    assert len(result[0][0][0]) == comps
    assert len(result[0]) == (len(treatment_list))
    assert len(result[0][0]) == len(times)


def test_CMTF():
    """ Test the CMTF method with fake data. """
    tensor = np.random.rand(5, 8, 23)
    matrix = np.random.rand(5, 8, 45)

    tFac = perform_CMTF(tensor, matrix, r=3)

    assert isinstance(tFac, tl.cp_tensor.CPTensor)
