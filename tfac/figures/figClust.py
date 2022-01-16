"""
This creates Figure 3 - plot clustergram for proteins.
"""
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tensorpack import perform_CP
from ..dataHelpers import proteinNames, form_tensor
from ..gene_module import ns_RNAseq_data, run_module, plot_modules, enrishment_analysis

def clustergram_proteins():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    tensor, _, _ = form_tensor()
    tFac = perform_CP(tensor, r=6)

    proteins = pd.DataFrame(tFac.factors[2][57367:], index = proteinNames(), columns=["comp1", "comp2", "comp3", "comp4", "comp5", "comp6"])
    decreased_proteins = proteins.loc[((-0.001>=proteins).any(1) | (proteins>= 0.001).any(1))]
    g = sns.clustermap(decreased_proteins, cmap="bwr", method="centroid", center=0, figsize=(14, 20))
    plt.savefig("output/clustergram_proteins.svg")

def gene_module_enrichm_plot():
    """ Plots the gene module analysis in heatmap, in the tfac/output folder as modules_v_components.svg """
    ns, data = ns_RNAseq_data()
    # running the module
    modules = run_module(ns, data)
    # prepare for saving and 
    names = sorted(list(set(modules.loc[:, 'module'])))
    module_expression = pd.DataFrame(
        index=names,
        columns=data.columns
    )
    for name in names:
        in_module = modules.loc[modules.loc[:, 'module'] == name]
        module = data.loc[in_module.index, :]
        module_expression.loc[name, :] = module.mean()

    # write the module expression into a csv file
    module_expression.to_csv("module_expression.csv")
    modules.to_csv("modules.csv")
    plot_modules(module_expression)
    enrishment_analysis(modules)
