"""
This creates Figure 4.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from ..Data_Mod import form_parafac2_tensor, ohsu_var
from ..tensor import OHSU_parafac2_decomp, R2Xparafac2, projections_to_factors
import matplotlib.pyplot as plt


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    row = 3
    col = 6
    ax, f = getSetup((24, 11), (row, col))
    return f
