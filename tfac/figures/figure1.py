"""
This creates Figure 1:
(a) tensor factorization cartoon
(b) tensor_svg.svg from the data folder
(c) R2X of the whole data, including gene expressions and protein levels.
"""
from .common import subplotLabel, getSetup
from tensorly import partial_svd
from tensorpack import Decomposition
from matplotlib.ticker import ScalarFormatter
from tensorpack.plot import *
from tensorpack.decomposition import impute_missing_mat
from ..dataHelpers import form_tensor


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((8, 3), (1, 3))

    tensor, _, _ = form_tensor()
    num_comps = 8
    # perform tensor decomposition from tensorpack with 8 components
    t = Decomposition(tensor, max_rr=num_comps+1)
    t.perform_tfac()
    tfacr2x(ax[0], t)

    # PCA for flattened data -- 42 (6 time x 7 treatment) x 498 (203 gene expr + 295 protein level)
    flatData = np.reshape(np.moveaxis(tensor, 2, 0), (-1, tensor.shape[2]))
    data = impute_missing_mat(flatData.T)
    flatData = data.T
    U, S, V = partial_svd(flatData, num_comps)
    scores = U @ np.diag(S)
    recon = [scores[:, :rr] @ V[:rr, :] for rr in range(num_comps)]

    PCAR2X = []
    for i in range(num_comps):
        vTop, vBottom = 0.0, 0.0
        mMask = np.isfinite(flatData)
        flatData = np.nan_to_num(flatData)
        vTop += np.linalg.norm(recon * mMask - flatData)**2.0
        vBottom += np.linalg.norm(flatData)**2.0
        PCAR2X.append(1.0 - vTop / vBottom)

    ax[1].plot(np.arange(1, num_comps+1), PCAR2X, ".", label="PCA")
    ax[1].set_ylabel("Normalized Explained Variance")
    ax[1].set_xlabel("Number of Components")
    ax[1].set_title("Flattened Matrix Fac R2X")
    ax[1].set_ylim(bottom=0.0)
    ax[1].legend()

    # data reduction
    ax[2].set_xscale("log", base=2)
    ax[2].plot(np.arange(1, num_comps+1), 1.0 - np.array(t.TR2X), ".", label="TFac")
    ax[2].plot(np.arange(1, num_comps+1), 1.0 - np.array(PCAR2X), ".", label="PCA")
    ax[2].set_ylabel("Normalized Unexplained Variance")
    ax[2].set_xlabel("Size of Reduced Data")
    ax[2].set_title("Data reduction, TFac vs. PCA")
    ax[2].set_ylim(bottom=0.0)
    ax[2].xaxis.set_major_formatter(ScalarFormatter())
    ax[2].legend()

    # Add subplot labels
    subplotLabel(ax)

    return f
