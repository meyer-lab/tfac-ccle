"""
Figure to show each components largest weights for MCF10A MEMA Data
"""

from .common import getSetup
from ..plotHelpers import plot_components_MEMA
from ..dataHelpers import Tensor_LINCS_MEMA

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((20, 10), (3, 5))

    MCF10A = Tensor_LINCS_MEMA("mcf10a_ssc_Level4.tsv.xz")

    plot_components_MEMA(MCF10A, ax)

    return f