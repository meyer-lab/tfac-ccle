""" factorize the MCF10A ECM data. """

from ..dataHelpers import import_LINCS_CycIF
from ..plotHelpers import plot_heatmaps
from .common import getSetup


def makeFigure():
    """ make heatmaps of factors when decomposed individually. """

    ax, f = getSetup((30, 15), (3, 1))
    tensor = import_LINCS_CycIF("mcf10a_ssc_Level4.tsv.xz")
    plot_heatmaps(tensor, ax)

    return f
