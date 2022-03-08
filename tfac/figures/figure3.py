""" factorize the ECM data. """
import pandas as pd
import numpy as np
import tensorly as tl
from tensorly.cp_tensor import cp_flip_sign
from tensorly.decomposition import parafac
from tensorpack.cmtf import cp_normalize
import seaborn as sns
from ..dataHelpers import import_LINCS_MEMA, reorder_table
from .common import getSetup


def makeFigure():
    """ make heatmaps of factors when decomposed individually. """
    ax, f = getSetup((30, 15), (3, 1))
    tensor, ligand, ecm, measurements = import_LINCS_MEMA("mcf10a_ssc_Level4.tsv.xz")
    fac = parafac(tensor, 5, n_iter_max=2000, linesearch=True, tol=1e-9)
    fac = cp_flip_sign(fac, 2)
    fac = cp_normalize(fac)

    facZero = pd.DataFrame(fac.factors[0], columns=[f"Cmp. {i}" for i in np.arange(1, fac.factors[0].shape[1] + 1)], index=ligand)
    facOne = pd.DataFrame(fac.factors[1], columns=[f"Cmp. {i}" for i in np.arange(1, fac.factors[1].shape[1] + 1)], index=ecm)
    facTwo = pd.DataFrame(fac.factors[2], columns=[f"Cmp. {i}" for i in np.arange(1, fac.factors[2].shape[1] + 1)], index=measurements)

    facZero = reorder_table(facZero)
    facOne = reorder_table(facOne)
    facTwo = reorder_table(facTwo)

    print("R2X: ", 1.0 - np.linalg.norm(tl.cp_to_tensor(fac) - tensor)**2.0 / np.linalg.norm(tensor)**2.0)

    sns.heatmap(facZero.T, ax=ax[0], cmap="PRGn", center=0)
    sns.heatmap(facOne.T, ax=ax[1], cmap="PRGn", center=0)
    sns.heatmap(facTwo.T, ax=ax[2], cmap="PRGn", center=0)
    return f
