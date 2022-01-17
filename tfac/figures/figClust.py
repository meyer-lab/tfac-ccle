"""
This creates Figure 3 - plot clustergram for proteins.
"""
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tensorpack import perform_CP
from ..dataHelpers import proteinNames, form_tensor

def clustergram_proteins_geneModules():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    tensor, _, _ = form_tensor()
    tFac = perform_CP(tensor, r=6)

    proteins = pd.DataFrame(tFac.factors[2][:295], index = proteinNames(), columns=["comp1", "comp2", "comp3", "comp4", "comp5", "comp6"])
    decreased_proteins = proteins.loc[((-0.001>=proteins).any(1) | (proteins>= 0.001).any(1))]
    g = sns.clustermap(decreased_proteins, cmap="bwr", method="centroid", center=0, figsize=(14, 20))
    plt.savefig("output/clustergram_proteins.svg")

    RNAseq = pd.read_csv(join(path_here, "tfac/data/ohsu/module_expression.csv"), delimiter=",")
    genes = pd.DataFrame(tFac.factors[2][295:], index = list(RNAseq["Unnamed: 0"]), columns=["comp1", "comp2", "comp3", "comp4", "comp5", "comp6"])
    decreased_genes = genes.loc[((-0.001>=genes).any(1) | (genes>= 0.001).any(1))]
    g = sns.clustermap(decreased_genes, cmap="bwr", method="centroid", center=0, figsize=(14, 20))
    plt.savefig("output/clustergram_geneModules.svg")
