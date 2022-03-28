"""
Figure to show each components largest weights for HMEC240L MEMA Data
"""
from .common import getSetup
from ..plotHelpers import plot_components_MEMA
from ..dataHelpers import import_LINCS_MEMA

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((20, 10), (3, 5))

    HMEC240 = import_LINCS_MEMA("hmec240l_ssc_Level4.tsv.xz")

    plot_components_MEMA(HMEC240, ax)

    return f 
