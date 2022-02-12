"""
This creates Figure 3 - plot clustergram for proteins.
"""
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tensorpack import perform_CP
import tensorly as tl
from tensorly.cp_tensor import cp_flip_sign
from tensorly.decomposition import parafac
from ..dataHelpers import import_LINCS_MEMA
from ..dataHelpers import proteinNames, form_tensor


def clustergram_proteins_geneModules():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    tensor, _, _ = form_tensor()
    tFac = perform_CP(tensor, 5, maxiter=2000)

    proteins = pd.DataFrame(tFac.factors[2][:295], index=proteinNames(), columns=["Cmp. 1", "Cmp. 2", "Cmp. 3", "Cmp. 4", "Cmp. 5"])
    tFac.normalize()

    decreased_proteins = proteins.loc[((-0.1 >= proteins).any(1) | (proteins >= 0.1).any(1))]
    g = sns.clustermap(decreased_proteins, cmap="PRGn", method="centroid", center=0, figsize=(8, 10), col_cluster=False)
    plt.savefig("output/clustergram_proteins.svg")

    RNAseq = pd.read_csv("tfac/data/ohsu/module_expression.csv", delimiter=",")
    genes = pd.DataFrame(tFac.factors[2][295:], index=list(RNAseq["Unnamed: 0"]), columns=["Cmp. 1", "Cmp. 2", "Cmp. 3", "Cmp. 4", "Cmp. 5"])
    decreased_genes = genes.loc[((-0.1 >= genes).any(1) | (genes >= 0.1).any(1))]
    g = sns.clustermap(decreased_genes, cmap="PRGn", method="centroid", center=0, figsize=(8, 14), col_cluster=False)
    plt.savefig("output/clustergram_geneModules.svg")

def cluster_mema():
    """ Plot the clustermap for the ECM data, separately when it is decomposed. """
    tensor, ligand, ecm, measurements = import_LINCS_MEMA()
    fac = parafac(tensor, 5, n_iter_max=2000, linesearch=True, tol=1e-8)
    fac.normalize()

    facZero = pd.DataFrame(fac.factors[0], columns=[f"{i}" for i in np.arange(1, fac.rank + 1)], index=ligand)
    decreased_ligand = facZero.loc[((-0.1 >= facZero).any(1) | (facZero >= 0.1).any(1))]
    facOne = pd.DataFrame(fac.factors[1], columns=[f"{i}" for i in np.arange(1, fac.rank + 1)], index=ecm)
    decreased_ecm = facOne.loc[((-0.1 >= facOne).any(1) | (facOne >= 0.1).any(1))]
    facTwo = pd.DataFrame(fac.factors[2], columns=[f"{i}" for i in np.arange(1, fac.rank + 1)], index=measurements)
    decreased_measurement = facTwo.loc[((-0.1 >= facTwo).any(1) | (facTwo >= 0.1).any(1))]

    print("R2X: ", 1.0 - np.linalg.norm(tl.cp_to_tensor(fac) - tensor)**2.0 / np.linalg.norm(tensor)**2.0)

    g = sns.clustermap(decreased_ligand, cmap="PRGn", method="centroid", center=0, figsize=(8, 12), col_cluster=False)
    plt.savefig("output/clustergram_ligand.svg")
    g = sns.clustermap(decreased_ecm, cmap="PRGn", method="centroid", center=0, figsize=(10, 16), col_cluster=False)
    plt.savefig("output/clustergram_ECM.svg")
    g = sns.clustermap(decreased_measurement, cmap="PRGn", method="centroid", center=0, figsize=(10, 18), col_cluster=False)
    plt.savefig("output/clustergram_measurements.svg")