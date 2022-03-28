""" factorize the HMEC240L ECM data. """

from ..dataHelpers import Tensor_LINCS_MEMA
from ..plotHelpers import plot_heatmaps
from .common import getSetup


def makeFigure():
    """ make heatmaps of factors when decomposed individually. """

    ax, f = getSetup((40, 10), (3, 1))
    tensor = Tensor_LINCS_MEMA("hmec240l_ssc_Level4.tsv.xz")
    plot_heatmaps(tensor, ax)

    return f
