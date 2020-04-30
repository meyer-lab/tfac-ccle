"""
This creates Figure 3 - All Tucker treatment component plots
"""
import numpy as np
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from .figure1 import treatmentPlot, timePlot, proteinPlot
from ..Data_Mod import form_tensor
from ..tensor import tucker_decomp, find_R2X_tucker

tensor, treatments, times = form_tensor()
results = tucker_decomp(tensor, (9, 9, 9))
factors = results[1]


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (2, 2))

    treatmentPlot(ax[1], factors[0], 1, 2, treatments)
    treatmentPlot(ax[2], factors[0], 2, 3, treatments)
    treatmentPlot(ax[3], factors[0], 3, 4, treatments)
    #timePlot(ax[2], factors[1])
    #proteinPlot(ax[3], factors[2], 1, 2)

    # Add subplot labels
    subplotLabel(ax)
    return f
