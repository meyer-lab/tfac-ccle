"""
This creates Figure 1:
(a) tensor factorization cartoon
(b) tensor_svg.svg from the data folder
(c) R2X of the whole data, including gene expressions and protein levels.
"""
from matplotlib.pyplot import axis
from .common import subplotLabel, getSetup
from tensorpack import Decomposition, perform_CP, calcR2X
from tensorpack.plot import *
from ..dataHelpers import import_LINCS_CCLE, import_LINCS_MEMA


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((8, 6), (2, 3))

    # ccle
    ccle, _, _ = import_LINCS_CCLE()
    # perform tensor decomposition from tensorpack with 8 components
    tc = Decomposition(ccle, max_rr=7)
    tc.perform_tfac()
    tc.perform_PCA(flattenon=2)

    tfacr2x(ax[0], tc)
    reduction(ax[1], tc)

    # mema
    mema, _, _, _ = import_LINCS_MEMA()
    tm = Decomposition(mema, max_rr=7)
    tm.perform_tfac()
    tm.perform_PCA(flattenon=2)

    tfacr2x(ax[3], tm)
    reduction(ax[4], tm)

    # Scaling factors for protein dataset
    R2Xs, scales = scaling(ccle, comps=5)

    labels = ['Protein','RNA','Total']
    for i in range(3):
        ax[2].plot(scales, R2Xs[i,:], label=labels[i])
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
    ax[3].set_title("Variance Explained by Tensor, MEMA")

    return f

def scaling(tensor, comps: int):
    """ Scaling function for proteins and gene expressions in CCLE dataset. """
    scales = np.power(4, [-4.0,-3.0,-2.0,-1.0,0,1,2,3,4])

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
        reconProt = recon[:,:,:295]
        reconRNA = recon[:,:,295:]
        recons = [reconProt,reconRNA,recon]

        # calculate R2X for proteins, RNA, whole Tensor
        for cc, rec in enumerate(recons):
            Top, Bottom = 0.0, 0.0
            tMask = np.isfinite(datas[cc])
            tIn = np.nan_to_num(datas[cc])
            Top += np.linalg.norm(rec * tMask - tIn) ** 2.0
            Bottom += np.linalg.norm(tIn) ** 2.0
            R2Xs[cc, c] = 1 - Top / Bottom
    return scales, R2Xs
