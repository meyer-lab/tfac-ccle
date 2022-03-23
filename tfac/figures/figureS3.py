"""
Figure to show each components largest weights for MCF10A MEMA Data
"""

from .common import getSetup
from ..plotHelpers import plot_components
from ..dataHelpers import import_LINCS_MEMA

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((20, 12), (3, 5))

    MCF10A = import_LINCS_MEMA("mcf10a_ssc_Level4.tsv.xz")

    plot_components(MCF10A, ax)

    return f
