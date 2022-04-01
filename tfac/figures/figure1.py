"""
This creates Figure 1:
"""
import numpy as np
from .common import subplotLabel, getSetup
from tensorly.decomposition import parafac
from tensorpack import Decomposition, perform_CP
from tensorpack.plot import tfacr2x, reduction
from ..dataHelpers import Tensor_LINCS_CCLE, Tensor_LINCS_MEMA, Tensor_LINCS_CycIF


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((9, 12), (5, 3))
    ax[5].axis("off")
    ax[8].axis("off")
    ax[11].axis("off")
    ax[14].axis("off")

    # ccle
    ccle = Tensor_LINCS_CCLE()
    # perform tensor decomposition from tensorpack with 8 components
    tc = Decomposition(ccle.to_numpy(), max_rr=7)
    tc.perform_tfac()
    tc.perform_PCA(flattenon=2)

    tfacr2x(ax[0], tc)
    reduction(ax[1], tc)
    ax[1].set_xlim((400, 4096))

    ppfac = lambda x, r: parafac(x, rank=r, n_iter_max=100, tol=1e-9, linesearch=True)

    # mema MCF10A
    MCF10A = Tensor_LINCS_MEMA("mcf10a_ssc_Level4.tsv.xz")
    tm = Decomposition(MCF10A.to_numpy(), max_rr=7, method=ppfac)
    tm.perform_tfac()
    tm.perform_PCA(flattenon=2)

    tfacr2x(ax[3], tm)
    reduction(ax[4], tm)
    ax[4].set_xlim((200, 8592))
    ax[4].set_xticks([256, 512, 1024, 2048, 4096, 8192])

    # mema HMEC240L
    HMEC240 = Tensor_LINCS_MEMA("hmec240l_ssc_Level4.tsv.xz")
    th = Decomposition(HMEC240.to_numpy(), max_rr=7, method=ppfac)
    th.perform_tfac()
    th.perform_PCA(flattenon=2)

    tfacr2x(ax[6], th)
    reduction(ax[7], th)
    ax[7].set_xlim((200, 8592))
    ax[7].set_xticks([256, 1024, 2048, 8192, 32768])

    # mema HMEC122L
    HMEC122 = Tensor_LINCS_MEMA("hmec122l_ssc_Level4.tsv.xz")
    th = Decomposition(HMEC122.to_numpy(), max_rr=7, method=ppfac)
    th.perform_tfac()
    th.perform_PCA(flattenon=2)

    tfacr2x(ax[9], th)
    reduction(ax[10], th)
    ax[10].set_xlim((200, 8592))
    ax[10].set_xticks([256, 1024, 2048, 8192, 32768])

    # mema CycIF
    CycIF = Tensor_LINCS_CycIF()
    th = Decomposition(CycIF.to_numpy(), max_rr=7, method=ppfac)
    th.perform_tfac()
    th.perform_PCA(flattenon=0)

    tfacr2x(ax[12], th)
    reduction(ax[13], th)
    ax[13].set_xlim((200, 8592))
    ax[13].set_xticks([256, 1024, 2048, 8192, 32768])

    # Scaling factors for protein dataset
    scales, R2Xs = scaling(ccle, comps=5)

    labels = ['Protein', 'RNA', 'Total']
    for i in range(3):
        ax[2].plot(scales, R2Xs[i, :], label=labels[i])
    ax[2].set_ylabel("R2X")
    ax[2].set_xlabel("Protein Variance Scaling Factor")
    ax[2].set_title("Variance explained of RNA and Protein")
    ax[2].legend()
    ax[2].set_xscale("log", base=4)
    ax[2].set_xticks([x for x in scales])
    ax[2].set_ylim(0, 1)

    labels = ['Protein', 'RNA', 'Total']
    for i in range(3):
        ax[2].plot(scales, R2Xs[i, :], label=labels[i])
    ax[2].set_ylabel("R2X")
    ax[2].set_xlabel("Protein Variance Scaling Factor")
    ax[2].set_title("Variance explained of RNA and Protein")
    ax[2].legend()
    ax[2].set_xscale("log", base=4)
    ax[2].set_xticks([x for x in scales])
    ax[2].set_ylim(0, 1)
    # Add subplot labels
    subplotLabel(ax)
    ax[0].set_title("Variance Explained by Tensor, CCLE")
    ax[3].set_title("Variance Explained by Tensor, MEMA, MCF10A")
    ax[6].set_title("Variance Explained by Tensor, MEMA, HMEC240L")
    ax[9].set_title("Variance Explained by Tensor, MEMA, HMEC122L")
    ax[12].set_title("Variance Explained by Tensor, MEMA, CycIF")

    return f


def scaling(tensor, comps: int):
    """ Scaling function for proteins and gene expressions in CCLE dataset. """
    scales = np.power(4, [-4.0, -3.0, -2.0, -1.0, 0, 1, 2, 3, 4])

    R2Xs = np.zeros((3, len(scales)))

    # Iterate through each scaling factor
    for c, s in enumerate(scales):
        # apply scaling to dataset (tensor * s)
        protData = tensor[:, :, :295] * s
        rnaData = tensor[:, :, 295:]
        newTensor = np.append(protData, rnaData, axis=2)
        datas = [protData, rnaData, newTensor]

        # perform cp and generate reconsturction
        tfac = perform_CP(newTensor, r=comps)
        recon = tfac.to_tensor()
        reconProt = recon[:, :, :295]
        reconRNA = recon[:, :, 295:]
        recons = [reconProt, reconRNA, recon]

        # calculate R2X for proteins, RNA, whole Tensor
        for cc, rec in enumerate(recons):
            Top, Bottom = 0.0, 0.0
            tMask = np.isfinite(datas[cc])
            tIn = np.nan_to_num(datas[cc])
            Top += np.linalg.norm(rec * tMask - tIn) ** 2.0
            Bottom += np.linalg.norm(tIn) ** 2.0
            R2Xs[cc, c] = 1 - Top / Bottom

    return scales, R2Xs
