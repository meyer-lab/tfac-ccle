"""
This creates Figure 1 - CP Decomposition Plots
"""
import numpy as np
import seaborn as sns
import pandas as pd
from .figureCommon import subplotLabel, getSetup
from ..tensor import cp_decomp, find_R2X_parafac, reorient_factors
from ..Data_Mod import form_tensor, z_score_tensor_bymeasure

tensor, treatments, times = form_tensor()
zscore_tensor_bymeasurement = z_score_tensor_bymeasure(tensor)
results = cp_decomp(zscore_tensor_bymeasurement, 13)
comps = reorient_factors(results[1])


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((7, 6), (2, 2))

    R2X_figure(ax[0], zscore_tensor_bymeasurement)
    treatmentPlot(ax[1], comps[0], treatments)
    timePlot(ax[2], comps[1], times)
    proteinPlot(ax[3], comps[2])

    # Add subplot labels
    subplotLabel(ax)

    return f


def R2X_figure(ax, input_tensor):
    '''Create Parafac R2X Figure'''
    R2X = np.zeros(14)
    nComps = range(1, len(R2X))
    for i in nComps:
        output = cp_decomp(input_tensor, i)
        R2X[i] = find_R2X_parafac(output, input_tensor)
    sns.scatterplot(np.arange(len(R2X)), R2X, ax=ax)
    ax.set_xlabel("Rank Decomposition")
    ax.set_ylabel("R2X")
    ax.set_title("CP Decomposition")
    ax.set_yticks([0, .2, .4, .6, .8, 1])


def treatmentPlot(ax, factors, senthue):
    '''Plot treatments (tensor axis 0) in factorization component space'''
    df = pd.DataFrame(factors).T
    markers = ['o', 's', 'X', 'h', '^', 'D', 'P']
    df.columns = senthue
    df["Components"] = range(1, 14)
    df = df.set_index('Components')
    sns.scatterplot(data=df, ax=ax, markers=markers, palette='bright', s=100)
    ax.set_xlabel('Component')
    ax.set_ylabel('Component Value')
    ax.set_title('Treatment Factors')  


def timePlot(ax, factors, senthue):
    '''Plot time points (tensor axis 1) in factorization component space'''
    df = pd.DataFrame(factors)
    columns = []
    for i in range(1, (factors.shape[1] + 1)):
        columns.append('Component ' + str(i))
    df.columns = columns
    df['Measurement Time'] = senthue
    df = df.set_index('Measurement Time')
    sns.lineplot(data=df, ax=ax, palette='tab20', dashes=False)
    ax.set_xlabel("Measurement Time")
    ax.set_ylabel('Component Value')
    ax.set_title('Time Factors')

def proteinPlot(ax, factors):
    '''Plot proteins (tensor axis 2) in factorization component space'''
    df = pd.DataFrame(factors)
    complist = range(1, (factors.shape[1] + 1))
    df.columns = complist
    sns.boxplot(data = df, ax = ax)
    ax.set_xlabel("Component")
    ax.set_ylabel('Component Value')
    ax.set_title('Protein Factors')


def setPlotLimits(axis, factors, r1, r2):
    '''Set appropriate limits for the borders of each component plot'''
    x = np.absolute(factors[:, r1 - 1])
    y = np.absolute(factors[:, r2 - 1])
    xlim = 1.1 * np.max(x)
    ylim = 1.1 * np.max(y)
    axis.set_xlim((-xlim, xlim))
    axis.set_ylim((-ylim, ylim))