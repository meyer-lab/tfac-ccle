"""
This creates Figure 1.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from .figureCommon import subplotLabel, getSetup
from ..Data_Mod import form_tensor
from ..tensor import cp_decomp, find_R2X_parafac


tensor, treatments, times = form_tensor()
output = cp_decomp(tensor, 2)
factors = output[1]

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
    R2X = np.zeros(14)
    nComps = range(1, len(R2X))
    for i in nComps:
        output = cp_decomp(tensor, i)
        R2X[i] = find_R2X_parafac(output, tensor)
    sns.scatterplot(np.arange(len(R2X)), R2X, ax=ax)
    ax.set_xlabel("Rank Decomposition")
    ax.set_ylabel("R2X")
    ax.set_title("CP Decomposition")


def treatmentPlot(ax, factors, r1, r2, senthue):
    '''Plot Treatment (tensor axis 0) in factorization component space'''
    sns.scatterplot(factors[:, r1 - 1], factors[:, r2 - 1], ax=ax, hue=senthue)
    ax.set_xlabel('Component ' + str(r1))
    ax.set_ylabel('Component ' + str(r2))
    ax.set_title('Treatment Factors')
    setPlotLimits(ax, factors, r1, r2)
    
def timePlot(ax, factors, r1, r2, senthue):
    '''Plot Cell Lines (tensor axis 0) in factorization component space'''
    sns.scatterplot(factors[:, r1 - 1], factors[:, r2 - 1], ax=ax, hue=senthue)
    ax.set_xlabel('Component ' + str(r1))
    ax.set_ylabel('Component ' + str(r2))
    ax.set_title('Time Factors')
    setPlotLimits(ax, factors, r1, r2)

def proteinPlot(ax, factors, r1, r2):
    '''Plot genes (tensor axis 1) in factorization component space'''
    sns.scatterplot(factors[:, r1 - 1], factors[:, r2 - 1], ax=ax)
    ax.set_xlabel('Component ' + str(r1))
    ax.set_ylabel('Component ' + str(r2))
    ax.set_title('Protein Factors')
    setPlotLimits(ax, factors, r1, r2)

def setPlotLimits(axis, factors, r1, r2):
    '''Set appropriate limits for the borders of each component plot'''
    x = np.absolute(factors[:, r1 - 1])
    y = np.absolute(factors[:, r2 - 1])
    xlim = 1.1 * np.max(x)
    ylim = 1.1 * np.max(y)
    axis.set_xlim((-xlim, xlim))
    axis.set_ylim((-ylim, ylim))
