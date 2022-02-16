import numpy as np
import tensorly as tl
from sklearn.decomposition import TruncatedSVD
from tensorpack.cmtf import calcR2X
from tensorly.cp_tensor import cp_flip_sign
from tensorly.decomposition import parafac
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from ..dataHelpers import import_LINCS_MEMA
from .common import getSetup

def makeFigure():
    "make plots of pca vs tfac"

    ax, f = getSetup((10, 5), (1, 2))
    tensor, ligand, ecm, measurements = import_LINCS_MEMA()
    comps = 5

    TR2X = []
    for comp in np.arange(1,comps+1):
        fac = parafac(tensor, comp, n_iter_max=2000, linesearch=True, tol=1e-8)

        fac = cp_flip_sign(fac, 2)
        fac.normalize()

        vTop, vBottom = 0.0, 0.0
        tMask = np.isfinite(tensor)
        tIn = np.nan_to_num(tensor)
        vTop += np.linalg.norm(fac.to_tensor() * tMask - tIn)**2.0
        vBottom += np.linalg.norm(tIn)**2.0

        TR2X.append(1 - vTop/vBottom)

    sizeT = [rr * sum(tensor.shape) for rr in np.arange(1,comps+1)]

    tensorShape = tensor.shape
    flatData = np.reshape(np.moveaxis(tensor, 2, 0), (tensorShape[2], -1))

    tsvd = TruncatedSVD(n_components=fac.rank)
    scores = tsvd.fit_transform(flatData)
    loadings = tsvd.components_

    recon = [scores[:, :rr] @ loadings[:rr, :] for rr in np.arange(1,comps+1)]
    PCAR2X = [calcR2X(c, mIn = flatData) for c in recon]
    sizePCA = [sum(flatData.shape) * rr for rr in np.arange(1,comps+1)]
    
    TR2X, PCAR2X, sizeTfac, sizePCA = np.asarray(TR2X), np.asarray(PCAR2X), sizeT, sizePCA
    ax[0].set_xscale("log", base=2)
    ax[0].plot(sizeTfac, 1.0 - TR2X, ".", label="TFac")
    ax[0].plot(sizePCA, 1.0 - PCAR2X, ".", label="PCA")
    ax[0].set_ylabel("Normalized Unexplained Variance")
    ax[0].set_xlabel("Size of Reduced Data")
    ax[0].set_title("Data reduction, TFac vs. PCA")
    ax[0].set_ylim(bottom=0.0)
    ax[0].xaxis.set_major_formatter(ScalarFormatter())
    ax[0].legend()

    ax[1].scatter(np.arange(1,comps+1), TR2X, s=10)
    ax[1].set_ylabel("Tensor Fac R2X")
    ax[1].set_xlabel("Number of Components")
    ax[1].set_title("Variance explained by tensor decomposition")
    ax[1].set_xticks(np.arange(1,comps+1))
    ax[1].set_xticklabels(np.arange(1,comps+1))
    ax[1].set_ylim(0, 1)
    ax[1].set_xlim(0.5, np.amax(comps) + 0.5)

    return f