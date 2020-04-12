"""
This creates Figure 1.
"""
import numpy as np
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from ..dataHelpers import cellLineNames
from ..Data_Mod import form_tensor
from ..tensor import perform_decomposition, find_R2X
import matplotlib.pyplot as plt

ts, junk1, junk2 = form_tensor()
factors = perform_decomposition(ts, 2)
def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (3, 2))

    ax[0].axis('off')  # blank out axes for cartoon
    ax[1].axis('off')
    cellLinePlot(ax[2], factors[0], 1,2, junk1)
    cellLinePlotTime(ax[3], factors[1], 1,2, junk2)
    cellLinePlot(ax[4], factors[2], 1,2, None)
    # Add subplot labels
    subplotLabel(ax)

    return f


def R2X_figure(ax):
    '''Create Parafac R2X Figure'''
    ax = sns.scatterplot(nComps, R2X, ax=ax)
    sns.despine(ax=ax)
    ax.set_xlabel("Rank Decomposition")
    ax.set_ylabel("R2X")
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_title("CP Decomposition")


def cellLinePlot(ax, factors, r1, r2, senthue):
    '''Plot Cell Lines (tensor axis 0) in factorization component space'''
    sns.scatterplot(factors[:, r1 - 1], factors[:, r2 - 1], ax=ax, hue=senthue)
    ax.set_xlabel('Component ' + str(r1))
    ax.set_ylabel('Component ' + str(r2))
    ax.set_title('Treatment Factors')
    
def cellLinePlotTime(ax, factors, r1, r2, senthue):
    '''Plot Cell Lines (tensor axis 0) in factorization component space'''
    sns.scatterplot(factors[:, r1 - 1], factors[:, r2 - 1], ax=ax, hue=senthue)
    ax.set_xlabel('Component ' + str(r1))
    ax.set_ylabel('Component ' + str(r2))
    ax.set_title('Time Factors')



def genePlot(ax, factors, r1, r2):
    '''Plot genes (tensor axis 1) in factorization component space'''
    sns.scatterplot(factors[:, r1 - 1], factors[:, r2 - 1], ax=ax)
    ax.set_xlabel('Component ' + str(r1))
    ax.set_ylabel('Component ' + str(r2))
    ax.set_title('Gene Factors')
    setPlotLimits(ax, factors, r1, r2)


def characPlot(ax, factors, r1, r2):
    '''Plot the measured genetic characteristics (tensor axis 2) in component space'''
    sns.scatterplot(factors[:, r1 - 1], factors[:, r2 - 1], ax=ax, style=['Gene Expression', 'Copy Number', 'Methylation'])
    ax.set_xlabel('Component ' + str(r1))
    ax.set_ylabel('Component ' + str(r2))
    ax.set_title('Genetic Characteristic Factors')
    setPlotLimits(ax, factors, r1, r2)


def setPlotLimits(axis, factors, r1, r2):
    '''Set appropriate limits for the borders of each component plot'''
    x = np.absolute(factors[:, r1 - 1])
    y = np.absolute(factors[:, r2 - 1])
    xlim = 1.1 * np.max(x)
    ylim = 1.1 * np.max(y)
    axis.set_xlim((-xlim, xlim))
    axis.set_ylim((-ylim, ylim))
    axis.axvline(color='black')
    axis.axhline(color='black')
