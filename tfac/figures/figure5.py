"""
This creates Figure 5. R2X for proteins with reverse projection reconstruction.
"""
import numpy as np
import seaborn as sns
from tensorly.metrics.regression import variance as tl_var
from .figureCommon import getSetup, subplotLabel
from ..dataHelpers import ohsu_data
from ..Data_Mod import form_tensor
from ..tensor import decomp_to_flipped_factors
from ..pseudoinvnorm import find_factors


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    row = 1
    col = 1
    ax, f = getSetup((4, 4), (row, col))
    protein_R2X(ax[0])
    subplotLabel(ax)
    return f


def protein_R2X(ax):
    """ data processing for protein """
    _, treatment_list, times = form_tensor()
    R2X = np.zeros(13)
    _, _, _, _, _, _, protein = ohsu_data()
    protein = protein.drop([
        "antibody",
        "BMP2_1",
        "BMP2_4",
        "BMP2_8",
        "EGF_1",
        "EGF_4",
        "EGF_8",
        "HGF_1",
        "HGF_4",
        "HGF_8",
        "IFNG_1",
        "IFNG_4",
        "IFNG_8",
        "OSM_1",
        "OSM_4",
        "OSM_8",
        "PBS_1",
        "PBS_4",
        "PBS_8",
        "TGFB_1",
        "TGFB_4",
        "TGFB_8",
    ], axis=1)
    colName = [
        "ctrl_0",
        "BMP2_24",
        "BMP2_48",
        "EGF_24",
        "EGF_48",
        "HGF_24",
        "HGF_48",
        "IFNg_24",
        "IFNg_48",
        "OSM_24",
        "OSM_48",
        "PBS_24",
        "PBS_48",
        "TGFb_24",
        "TGFb_48",
    ]
    ctrl = protein["ctrl_0"]
    for x, i in enumerate(treatment_list):
        protein.insert(x, i + "_0", ctrl, allow_duplicates=True)

    protein.rename(
        columns={
            "TGFB_1": "TGFb_1",
            "TGFB_4": "TGFb_4",
            "TGFB_8": "TGFb_8",
            "TGFB_24": "TGFb_24",
            "TGFB_48": "TGFb_48",
            "IFNG_1": "IFNg_1",
            "IFNG_4": "IFNg_4",
            "IFNG_8": "IFNg_8",
            "IFNG_24": "IFNg_24",
            "IFNG_48": "IFNg_48",
        },
        inplace=True,
    )
    protein = protein.reindex(colName, axis=1)
    # R2X calculation
    for i in range(1, 13):
        result, _, _ = decomp_to_flipped_factors(i)
        P, X, W = find_factors(result, protein, treatment_list, times)
        protein_redone = np.matmul(W.T, P)
        R2X[i] = 1 - tl_var(protein_redone - X) / tl_var(protein.to_numpy())
    sns.scatterplot(np.arange(len(R2X)), R2X, ax=ax)
    ax.set_xlabel("Rank Decomposition")
    ax.set_ylabel("Protein R2X")
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
