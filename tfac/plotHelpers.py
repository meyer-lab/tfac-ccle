""" Plotting helpers. """

import pandas as pd
import numpy as np
import tensorly as tl
from tensorly.cp_tensor import cp_flip_sign
from tensorly.decomposition import parafac
from tensorpack.cmtf import cp_normalize
import seaborn as sns
from .dataHelpers import reorder_table


def plot_heatmaps(tensor, ax):
    """ Plots all the components across all factors for the MEMA datasets. """
    fac = parafac(tensor.to_numpy(), 5, n_iter_max=2000, linesearch=True, tol=1e-9) # tensor is xarray type
    fac = cp_flip_sign(fac, 2)
    fac = cp_normalize(fac)

    labels = [f"Cmp. {i}" for i in np.arange(1, fac.factors[0].shape[1] + 1)]
    print("R2X: ", 1.0 - np.linalg.norm(tl.cp_to_tensor(fac) - tensor)**2.0 / np.linalg.norm(tensor)**2.0)

    for ii in range(tensor.ndim):
        facZero = pd.DataFrame(fac.factors[ii], columns=labels, index=tensor.coords[tensor.dims[ii]])
        facZero = reorder_table(facZero)
        sns.heatmap(facZero.T, ax=ax[ii], cmap="PRGn", center=0)

def plot_components_MEMA(tensor, ax):
    """ Plots most significant components separately in the supplementary figures. """
    tFac = parafac(tensor.to_numpy(), 5, n_iter_max=2000, linesearch=True, tol=1e-9) # tensor is xarray type
    tFac = cp_flip_sign(tFac, 2)
    tFac = cp_normalize(tFac)
    labels = [f"Cmp. {i}" for i in np.arange(1, tFac.factors[0].shape[1] + 1)]

    k = 0
    for ii in range(tensor.ndim):
        facZero = pd.DataFrame(tFac.factors[ii], columns=labels, index=tensor.coords[tensor.dims[ii]])
        facZero = reorder_table(facZero)

        for c, col in enumerate(facZero.keys()):
            feature = facZero[[col]]

            feature_l_ind = feature.abs().nlargest(10, col).index

            g0 = sns.heatmap(feature.loc[feature_l_ind].sort_values([col]), ax=ax[c + k], cmap="PRGn", center=0, vmin=-1, vmax=1)
            g0.set_title(f"Proteins, {col}")
        k += 5
