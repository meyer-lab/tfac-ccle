"""
This creates Figure 3 - plot clustergram for proteins.
"""
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tensorpack import perform_CP
import tensorly as tl
from tensorly.cp_tensor import cp_flip_sign, CPTensor
from tensorly.decomposition import parafac
from ..dataHelpers import proteinNames, import_LINCS_CCLE, import_LINCS_MEMA


def clustergram_proteins_geneModules():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    tensor, _, _ = import_LINCS_CCLE()
    tFac = perform_CP(tensor, 5, maxiter=2000)

    proteins = pd.DataFrame(tFac.factors[2][:295], index=proteinNames(), columns=["Cmp. 1", "Cmp. 2", "Cmp. 3", "Cmp. 4", "Cmp. 5"])

    decreased_proteins = proteins.loc[((-0.1 >= proteins).any(1) | (proteins >= 0.1).any(1))]
    g = sns.clustermap(decreased_proteins, cmap="PRGn", method="centroid", center=0, figsize=(8, 18), col_cluster=False)
    plt.savefig("output/clustergram_proteins.svg")

    RNAseq = pd.read_csv("tfac/data/ohsu/module_expression.csv", delimiter=",")
    genes = pd.DataFrame(tFac.factors[2][295:], index=list(RNAseq["Unnamed: 0"]), columns=["Cmp. 1", "Cmp. 2", "Cmp. 3", "Cmp. 4", "Cmp. 5"])
    decreased_genes = genes.loc[((-0.3 >= genes).any(1) | (genes >= 0.3).any(1))]
    g = sns.clustermap(decreased_genes, cmap="PRGn", method="centroid", center=0, figsize=(8, 14), col_cluster=False)
    plt.savefig("output/clustergram_geneModules.svg")


def cp_normalize(cp_tensor):
    weights, factors = cp_tensor
    rank = cp_tensor.rank
    
    if weights is None:
        weights = np.ones(rank)
    
    normalized_factors = []
    for i, factor in enumerate(factors):
        if i == 0:
            factor = factor*weights
            weights = np.ones(rank)
            
        scales = np.linalg.norm(factor, ord=np.inf, axis=0)
        weights = weights*scales
        normalized_factors.append(factor / np.reshape(scales, (1, -1)))

    return CPTensor((weights, normalized_factors))


def cluster_mema(cellline, num):
    """ Plot the clustermap for the ECM data, separately when it is decomposed. """
    tensor, ligand, ecm, measurements = import_LINCS_MEMA(cellline + "_ssc_Level4.tsv.xz")
    fac = parafac(tensor, num, n_iter_max=2000, linesearch=True, tol=1e-12)
    fac = cp_normalize(fac)
    fac = cp_flip_sign(fac, mode=2)

    facZero = pd.DataFrame(fac.factors[0], columns=[f"{i}" for i in np.arange(1, fac.rank + 1)], index=ligand)
    decreased_ligand = facZero.loc[((-0.1 >= facZero).any(1) | (facZero >= 0.1).any(1))]
    facOne = pd.DataFrame(fac.factors[1], columns=[f"{i}" for i in np.arange(1, fac.rank + 1)], index=ecm)
    decreased_ecm = facOne.loc[((-0.1 >= facOne).any(1) | (facOne >= 0.1).any(1))]
    facTwo = pd.DataFrame(fac.factors[2], columns=[f"{i}" for i in np.arange(1, fac.rank + 1)], index=measurements)
    decreased_measurement = facTwo.loc[((-0.1 >= facTwo).any(1) | (facTwo >= 0.1).any(1))]

    print("R2X: ", 1.0 - np.linalg.norm(tl.cp_to_tensor(fac) - tensor)**2.0 / np.linalg.norm(tensor)**2.0)

    g = sns.clustermap(decreased_ligand, cmap="PRGn", method="centroid", center=0, figsize=(8, 25), col_cluster=False)
    plt.savefig("output/ligands_" + cellline + ".svg")
    g = sns.clustermap(decreased_ecm, cmap="PRGn", method="centroid", center=0, figsize=(8, 16), col_cluster=False)
    plt.savefig("output/ECM_" + cellline + ".svg")
    g = sns.clustermap(decreased_measurement, cmap="PRGn", method="centroid", center=0, figsize=(8, 28), col_cluster=False)
    plt.savefig("output/measurements_" + cellline + ".svg")
