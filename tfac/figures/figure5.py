"""
This creates Figure 5.
"""

from .figureCommon import subplotLabel, getSetup


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((8, 8), (1, 1))

    # Add subplot labels
    subplotLabel(ax)

    return f