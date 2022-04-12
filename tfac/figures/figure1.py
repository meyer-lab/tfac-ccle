"""
This creates Figure 1:
"""
import numpy as np
from matplotlib.ticker import ScalarFormatter
from .common import subplotLabel, getSetup
from tensorly.decomposition import parafac
from tensorpack import Decomposition, perform_CP
from tensorpack.plot import tfacr2x, reduction
from ..dataHelpers import Tensor_LINCS_CCLE, Tensor_LINCS_MEMA, Tensor_LINCS_CycIF
from ..tucker import tucker_decomp, error_vs_size


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((10, 8), (3, 3))

    # ccle
    ccle = Tensor_LINCS_CCLE()
    # perform parafac
    tc = Decomposition(ccle.to_numpy(), max_rr=8)
    tc.perform_tfac()
    tc.perform_PCA(flattenon=2)

    tfacr2x(ax[0], tc)
    reduction(ax[1], tc)
    ax[1].set_xscale("log", base=2)
    ax[8].axis('off')

    ppfac = lambda x, r: parafac(x, rank=r, n_iter_max=100, tol=1e-9, linesearch=True)

    ### MEMA MCF10A
    MCF10A = Tensor_LINCS_MEMA("mcf10a_ssc_Level4.tsv.xz")
    tm = Decomposition(MCF10A.to_numpy(), max_rr=8, method=ppfac)
    tm.perform_tfac()
    tm.perform_PCA(flattenon=0)

    # R2X plot
    comps = tm.rrs
    ax[3].scatter(comps, tm.TR2X)
    ax[3].set_ylabel("Explained Variance")
    ax[3].set_xlabel("Number of Components")
    ax[3].set_title("R2X Tensor Decomp")
    ax[3].set_xticks([x for x in comps])
    ax[3].set_xticklabels([x for x in comps])
    ax[3].set_ylim(0, 1)
    ax[3].set_xlim(0.5, np.amax(comps) + 0.5)

    # reduction plot
    CPR2X, PCAR2X, sizeTfac, sizePCA = np.asarray(tm.TR2X), np.asarray(tm.PCAR2X), tm.sizeT, tm.sizePCA
    lin1, = ax[4].plot(sizeTfac, 1.0 - CPR2X, "*", alpha=0.8, color='C0')
    lin2, = ax[4].plot(sizePCA, 1.0 - PCAR2X, "^", alpha=0.8, color='C0')
    ax[4].set_xscale("log", base=2)
    ax[4].set_ylabel("Normalized Unexplained Variance")
    ax[4].set_xlabel("Size of Reduced Data")
    ax[4].set_title("Data reduction, TFac vs. PCA")
    ax[4].set_ylim(bottom=0.0)
    ax[4].xaxis.set_major_formatter(ScalarFormatter())

    # tucker
    errors, ranks = tucker_decomp(MCF10A, 25)
    sizes = error_vs_size(MCF10A, ranks)
    ax[5].scatter(sizes, errors)
    ax[5].set_ylim((0.0, 1.0))
    ax[5].set_title('Data reduction, Tucker')
    ax[5].set_ylabel('Normalized Unexplained Variance')
    ax[5].set_xlabel('Size of Reduced Data')

    ### MEMA HMEC240L
    HMEC240 = Tensor_LINCS_MEMA("hmec240l_ssc_Level4.tsv.xz")
    th = Decomposition(HMEC240.to_numpy(), max_rr=8, method=ppfac)
    th.perform_tfac()
    th.perform_PCA(flattenon=0)

    # R2X
    ax[3].scatter(th.rrs, th.TR2X)

    # Data Reduction
    CPR2X, PCAR2X, sizeTfac, sizePCA = np.asarray(th.TR2X), np.asarray(th.PCAR2X), th.sizeT, th.sizePCA
    lin3, = ax[4].plot(sizeTfac, 1.0 - CPR2X, "*", label="TFac", alpha=0.8, color='C1')
    lin4, = ax[4].plot(sizePCA, 1.0 - PCAR2X, "^", label="PCA", alpha=0.8, color='C1')

    # tucker
    errors, ranks = tucker_decomp(HMEC240, 25)
    sizes = error_vs_size(HMEC240, ranks)
    ax[5].scatter(sizes, errors)

    ### MEMA HMEC122L
    HMEC122 = Tensor_LINCS_MEMA("hmec122l_ssc_Level4.tsv.xz")
    th = Decomposition(HMEC122.to_numpy(), max_rr=8, method=ppfac)
    th.perform_tfac()
    th.perform_PCA(flattenon=0)

    # R2X
    ax[3].scatter(th.rrs, th.TR2X)
    ax[3].legend(['MCF10A', 'HMEC240L', 'HMEC122L'])

    # Data Reduction
    CPR2X, PCAR2X, sizeTfac, sizePCA = np.asarray(th.TR2X), np.asarray(th.PCAR2X), th.sizeT, th.sizePCA
    lin5, = ax[4].plot(sizeTfac, 1.0 - CPR2X, "*", label="TFac", alpha=0.8, color='C2')
    lin6, = ax[4].plot(sizePCA, 1.0 - PCAR2X, "^", label="PCA", alpha=0.8, color='C2')
    line1, = ax[4].plot([],[],marker='*', color='k', ls="none")
    line2, = ax[4].plot([],[],marker='^', color='k', ls="none")
    ax[4].legend([lin1, lin3, lin5, line1, line2], ['MCF10A', 'HMEC240L', 'HMEC122L', 'TFac', 'PCA'])
    ax[4].set_xscale("log", base=2)

    # tucker
    errors, ranks = tucker_decomp(HMEC122, 25)
    sizes = error_vs_size(HMEC122, ranks)
    ax[5].scatter(sizes, errors)
    ax[5].legend(['MCF10A', 'HMEC240L', 'HMEC122L'])
    ax[5].set_xscale("log", base=2)

    ### MEMA CycIF
    CycIF = Tensor_LINCS_CycIF()
    th = Decomposition(CycIF.to_numpy(), max_rr=8, method=ppfac)
    th.perform_tfac()
    th.perform_PCA(flattenon=0)

    tfacr2x(ax[6], th)
    reduction(ax[7], th)
    ax[7].set_xscale("log", base=2)

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
