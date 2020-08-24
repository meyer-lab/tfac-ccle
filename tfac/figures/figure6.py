"""
This creates Figure 6. Includes OHSU R2X, treatments by components, and outliers by data slice (experiment) plots.
"""
import matplotlib.pyplot as plt
import seaborn as sns
from .figureCommon import getSetup, subplotLabel

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    row = 1
    col = 1
    ax, f = getSetup((10, 10), (row, col))
    subplotLabel(ax)
    return f
    