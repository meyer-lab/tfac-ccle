"""
This creates Figure 1.
"""
import numpy as np
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from ..dataHelpers import cellLineNames
from ..Data_Mod import form_tensor
from ..tensor import perform_decomposition, find_R2X, find_R2X_nnp
import matplotlib.pyplot as plt


tensor, treatments, times = form_tensor()
factors = perform_decomposition(tensor, 2)

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (3, 2))

    ax[0].axis('off')  # blank out axes for cartoon
    R2X_figure(ax[1])
    treatmentPlot(ax[2], factors[0], 1, 2, treatments)
    timePlot(ax[3], factors[1], 1, 2, times)
    proteinPlot(ax[4], factors[2], 1, 2)

    # Add subplot labels
    subplotLabel(ax)

    return f


def R2X_figure(ax):
    '''Create Parafac R2X Figure'''
    R2X = []
    nComps = range(1,11)
    for i in nComps:
        R2X.append(find_R2X_nnp(form_tensor()[0],i))
    ax = sns.scatterplot(nComps, R2X, ax=ax)
    ax.set_xlabel("Rank Decomposition")
    ax.set_ylabel("R2X")
    ax.set_title("CP Decomposition")


def treatmentPlot(ax, factors, r1, r2, senthue):
    '''Plot Treatment (tensor axis 0) in factorization component space'''
    sns.scatterplot(factors[:, r1 - 1], factors[:, r2 - 1], ax=ax, hue=senthue)
    ax.set_xlabel('Component ' + str(r1))
    ax.set_ylabel('Component ' + str(r2))
    ax.set_title('Treatment Factors')
    
def timePlot(ax, factors, r1, r2, senthue):
    '''Plot Cell Lines (tensor axis 0) in factorization component space'''
    sns.scatterplot(factors[:, r1 - 1], factors[:, r2 - 1], ax=ax, hue=senthue)
    ax.set_xlabel('Component ' + str(r1))
    ax.set_ylabel('Component ' + str(r2))
    ax.set_title('Time Factors')



def proteinPlot(ax, factors, r1, r2):
    '''Plot genes (tensor axis 1) in factorization component space'''
    sns.scatterplot(factors[:, r1 - 1], factors[:, r2 - 1], ax=ax)
    ax.set_xlabel('Component ' + str(r1))
    ax.set_ylabel('Component ' + str(r2))
    ax.set_title('Protein Factors')


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
