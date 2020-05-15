"""
This creates Figure 4.
"""
from .figureCommon import subplotLabel, getSetup
from .figure1 import proteinScatterPlot


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((15, 20), (7, 4))

    counter = 0
    for i in range(8):
        for j in range(i + 1, 8):
            proteinScatterPlot(ax[counter], comps[2], i+1, j+1)
            counter += 1
    
    # Add subplot labels
    subplotLabel(ax)

    return f
