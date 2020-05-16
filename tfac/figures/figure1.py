"""
This creates Figure 1 - CP Decomposition Plots
"""
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from .figureCommon import subplotLabel, getSetup
from ..tensor import cp_decomp, find_R2X_parafac, reorient_factors
from ..Data_Mod import form_tensor, z_score_tensor_bymeasure, z_score_tensor_byprotein

tensor, treatments, times = form_tensor()
tensor_z = z_score_tensor_byprotein(tensor)
results = cp_decomp(tensor_z, 8)
comps = reorient_factors(results[1])


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((12, 12), (2, 2))

    R2X_figure(ax[0], tensor_z)
    treatmentPlot(ax[1], comps[0], treatments)
    timePlot(ax[2], comps[1], times)
    proteinBoxPlot(ax[3], comps[2])

    # Add subplot labels
    subplotLabel(ax)

    return f


def R2X_figure(ax, input_tensor):
    '''Create Parafac R2X Figure'''
    R2X = np.zeros(13)
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
    df["Components"] = range(1, factors.shape[1] + 1)
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

def proteinBoxPlot(ax, factors):
    '''Plot proteins (tensor axis 2) in factorization component space'''
    df = pd.DataFrame(factors)
    complist = range(1, (factors.shape[1] + 1))
    df.columns = complist
    sns.boxplot(data=df, ax=ax)
    ax.set_xlabel("Component")
    ax.set_ylabel('Component Value')
    ax.set_title('Protein Factors')

def proteinScatterPlot(ax, factors, r1, r2):
    '''Plot compared proteins (tensor axis 2) in factorization component space'''
    sns.scatterplot(factors[:, r1 - 1], factors[:, r2 - 1], ax=ax)
    ax.set_xlabel('Component ' + str(r1))
    ax.set_ylabel('Component ' + str(r2))
    ax.set_title('Protein Factors')

def setPlotLimits(axis, factors, r1, r2):
    '''Set appropriate limits for the borders of each component plot'''
    x = np.absolute(factors[:, r1 - 1])
    y = np.absolute(factors[:, r2 - 1])
    xlim = 1.1 * np.max(x)
    ylim = 1.1 * np.max(y)
    axis.set_xlim((-xlim, xlim))
    axis.set_ylim((-ylim, ylim))

def separate_treatmentPlot(factors, senthue):
    '''Plot treatments (tensor axis 0) in factorization component space'''
    df = pd.DataFrame(factors)
    df.columns = range(1, factors.shape[1] + 1)
    df["Treatment"] = senthue
    df = pd.melt(df, "Treatment")
    df.columns = ["Treatment", "Component", "Value"]
    grid = sns.FacetGrid(df, col="Treatment", hue="Treatment", col_wrap=4, height=4, sharex=False)
    grid.map(plt.plot, "Component", "Value")
    for i in range(1, factors.shape[1] + 1):
        grid.map(plt.axvline, x=i, ls=":", c=".7")
    for i in np.arange(-.8, 1, .2):
        grid.map(plt.axhline, y=i, ls=":", c=".7")
    grid.map(plt.axhline, y=0, ls="-", c=".5")
    grid.set(xticks=np.arange(1, factors.shape[1] + 1))
    grid.set(yticks=np.arange(-.8, 1, .2))
    for i in range(factors.shape[0]):
        grid.axes[i].set_xlabel('Components')
    grid.fig.tight_layout(w_pad=1)
