"""
This creates Figure 4 - All Tucker protein component plots
"""
import numpy as np
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from .figure1 import treatmentPlot, timePlot, proteinPlot
from ..Data_Mod import form_tensor
from ..tensor import tucker_decomp

tensor, treatments, times = form_tensor()
results = tucker_decomp(tensor, (9, 9, 9))
factors1 = results[1]


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (3, 3))

    proteinPlot(ax[0], factors1[2], 1, 2)
    proteinPlot(ax[1], factors1[2], 2, 3)
    proteinPlot(ax[2], factors1[2], 3, 4)
    proteinPlot(ax[3], factors1[2], 4, 5)
    proteinPlot(ax[4], factors1[2], 5, 6)
    proteinPlot(ax[5], factors1[2], 6, 7)
    proteinPlot(ax[6], factors1[2], 7, 8)
    proteinPlot(ax[7], factors1[2], 8, 9)
    
    # Add subplot labels
    subplotLabel(ax)
    return f
