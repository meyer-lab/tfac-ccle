"""
This creates Figure 2.
"""
import seaborn as sns
from tensorly.decomposition import parafac
from .figureCommon import subplotLabel, getSetup
from .figure1 import treatmentPlot, timePlot, proteinPlot, setPlotLimits
from ..Data_Mod import form_tensor
from ..regression import KFoldCV
from ..tensor import find_R2X
import numpy as np

tensor, treatments, times = form_tensor()
_, factors = parafac(tensor, 12, orthogonalise=True)
temp1 = np.ndarray(shape=(6,12), dtype=float, order='C')

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (2, 2))

    R2X_figure(ax[0])
    treatmentPlot(ax[1], factors[0], 1, 2, treatments)
    timePlot(ax[2], factors[1], 1, 2, times)
    proteinPlot(ax[3], factors[2], 1, 2)

    # Add subplot labels
    subplotLabel(ax)
    return f

def R2X_figure(ax):
    '''Create Parafac R2X Figure'''
    R2X = []
    nComps = range(1, 14)
    for i in nComps:
        R2X.append(find_R2X(form_tensor()[0], i))
    ax = sns.scatterplot(nComps, R2X, ax=ax)
    ax.set_xlabel("Rank Decomposition")
    ax.set_ylabel("R2X")
    ax.set_title("CP Decomposition")
