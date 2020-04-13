"""
This creates Figure 2.
"""
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from .figure1 import treatmentPlot, timePlot, proteinPlot
from ..Data_Mod import form_tensor
from ..regression import KFoldCV
from ..tensor import find_R2X_parafac


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (2, 2))

    ax[0].axis('off')  # blank out first axis for cartoon
    R2X_figure(ax[1])

    # Add subplot labels
    subplotLabel(ax)
    return f

def R2X_figure(ax):
    '''Create Parafac R2X Figure'''
    R2X = []
    nComps = range(1,11)
    for i in nComps:
        R2X.append(find_R2X_parafac(form_tensor()[0], i))
    ax = sns.scatterplot(nComps, R2X, ax=ax)
    ax.set_xlabel("Rank Decomposition")
    ax.set_ylabel("R2X")
    ax.set_title("CP Decomposition")


#### FROM ORIGINAL PROJECT #################################################################################

def predVsActual(ax, x, y, reg):
    '''Predicted vs Actual plotting function for regression'''
    _, predicted, actual = KFoldCV(x, y, reg)
    sns.scatterplot(actual, predicted, color='darkslategrey', ax=ax)
    sns.despine()
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Predicted vs Actual ' + reg)
