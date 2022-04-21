""" factorize the HMEC122L ECM data. """

from ..dataHelpers import Tensor_LINCS_MEMA, process_RA_Tensor
from ..plotHelpers import plot_heatmaps
from .common import getSetup


def makeFigure():
    """ make heatmaps of factors when decomposed individually. """

    ax, f = getSetup((40, 10), (3, 1))
    process_RA_Tensor()
    #tensor = Tensor_LINCS_MEMA("hmec122l_ssc_Level4.tsv.xz")
    #plot_heatmaps(tensor, ax)

    return f
