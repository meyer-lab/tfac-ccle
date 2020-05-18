"""
This creates Figure 2 - Tucker Decomposition Plots
"""
import numpy as np
import seaborn as sns
from tensorly.decomposition import parafac
from .figureCommon import subplotLabel, getSetup
from .figure1 import treatmentPlot, timePlot, proteinBoxPlot, setPlotLimits, R2X_figure
from ..Data_Mod import form_tensor
from ..regression import KFoldCV
from ..tensor import find_R2X_parafac

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (3, 3))

    # Add subplot labels
    subplotLabel(ax)
    return f


#### FROM ORIGINAL PROJECT #################################################################################

def predVsActual(ax, x, y, reg):
    '''Predicted vs Actual plotting function for regression'''
    _, predicted, actual = KFoldCV(x, y, reg)
    sns.scatterplot(actual, predicted, color='darkslategrey', ax=ax)
    sns.despine()
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Predicted vs Actual ' + reg)
