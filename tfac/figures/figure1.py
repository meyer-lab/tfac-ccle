"""
This creates Figure 1.
"""
import numpy as np
import matplotlib as plt
import tensorly as tl
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from ..tensor import calc_R2X_parafac, perform_parafac

### Data Tensor Variable Gets Declared Here
tensor = tl.random.random_kruskal((50, 50, 50), rank=20, full=True)
factors = perform_parafac(tensor, 4)


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((7.5, 6), (3, 4))

    ax[0].axis('off')  # blank out first axis for cartoon
    ax[1].axis('off')
    R2X_figure(ax[2], tensor)
    cellLinePlot(ax[4], factors[0], 1, 2)
    cellLinePlot(ax[5], factors[0], 3, 4)

    # Add subplot labels
    subplotLabel(ax)

    return f

def R2X_figure(ax, tens):
    '''Create Parafac R2X Figure'''
    x_axis = np.arange(9)
    R2X = np.zeros(9)
    for i in range (1, 9):
        R2X[i] = calc_R2X_parafac(tens, i)
    ax.scatter(x_axis, R2X)
    ax.set_xlabel('Decomposition Rank')
    ax.set_ylabel('R2X')
    ax.set_title('PARAFAC')
    ax.set_yticks([0, .2, .4, .6, .8, 1.0])
    ax.set_xticks(x_axis)
    
def cellLinePlot(ax, factors, r1, r2):
    '''Plot Cell Lines (tensor axis 0) in factorization component space'''
    sns.scatterplot(factors[r1 - 1], factors[r2 - 1], ax=ax)
    ax.set_xlabel('Component ' + str(r1))
    ax.set_ylabel('Component ' + str(r2))
    ax.set_title('Cell Line Factors')
    
def genePlot(ax, factors, r1, r2):
    '''Plot genes (tensor axis 1) in factorization component space'''
    sns.scatterplot(factors[r1 - 1], factors[r2 - 1], ax=ax)
    ax.set_xlabel('Component ' + str(r1))
    ax.set_ylabel('Component ' + str(r2))
    ax.set_title('Gene Factors')
