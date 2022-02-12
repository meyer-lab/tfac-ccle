""" factorize the ECM data. """
import pandas as pd
import numpy as np
import tensorly as tl
from tensorly.cp_tensor import cp_flip_sign
from tensorly.decomposition import parafac
import matplotlib.pyplot as plt
import seaborn as sns
from ..dataHelpers import import_LINCS_MEMA
from .common import getSetup

def makeFigure():
    """ make heatmaps of factors when decomposed individually. """
    ax, f = getSetup((20, 10), (3, 1))
    tensor, ligand, ecm, measurements = import_LINCS_MEMA()
    fac = parafac(tensor, 5, n_iter_max=2000, linesearch=True, tol=1e-8)

    fac = cp_flip_sign(fac, 2)
    fac.normalize()

    facZero = pd.DataFrame(fac.factors[0], columns=[f"{i}" for i in np.arange(1, fac.rank + 1)], index=ligand)
    facOne = pd.DataFrame(fac.factors[1], columns=[f"{i}" for i in np.arange(1, fac.rank + 1)], index=ecm)
    facTwo = pd.DataFrame(fac.factors[2], columns=[f"{i}" for i in np.arange(1, fac.rank + 1)], index=measurements)

    print("R2X: ", 1.0 - np.linalg.norm(tl.cp_to_tensor(fac) - tensor)**2.0 / np.linalg.norm(tensor)**2.0)

    sns.heatmap(facZero.T, ax=ax[0], center=0)
    sns.heatmap(facOne.T, ax=ax[1], center=0)
    sns.heatmap(facTwo.T, ax=ax[2], center=0)
    return f

