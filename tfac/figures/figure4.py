"""
This creates Figure 4. Gene expression R2X with flattened matrix dimension reconstruction.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorly.metrics.regression import variance as tl_var
from .figureCommon import subplotLabel, getSetup
from ..Data_Mod import form_tensor
from ..tensor import decomp_to_flipped_factors
from ..pseudoinvnorm import find_factors
from ..dataHelpers import proteinNames


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    row = 1
    col = 1
    ax, f = getSetup((70, 10), (row, col))
    makeProteinListDistribution(ax[0])
    subplotLabel(ax)
    return f
    

def makeProteinListDistribution(axis):
    components = 5
    results, treatmentList, times = decomp_to_flipped_factors(components)

    proteinFactors = results[1][0]
    proteinList = proteinNames()

    comp1Vals = []
    comp2Vals = []
    comp3Vals = []
    comp4Vals = []
    comp5Vals = []

    for compVals in proteinFactors:
        comp1Vals.append(compVals[0])
        comp2Vals.append(compVals[1])
        comp3Vals.append(compVals[2])
        comp4Vals.append(compVals[3])
        comp5Vals.append(compVals[4])

    plt.figure(figsize=(70,10))



    plt.scatter(proteinList, comp1Vals, label = 'Component 1', ax=axis)
    plt.scatter(proteinList, comp2Vals, label = 'Component 2', ax=axis)
    plt.scatter(proteinList, comp3Vals, label = 'Component 3',ax=axis)
    plt.scatter(proteinList, comp4Vals, label = 'Component 4', ax=axis)
    plt.scatter(proteinList, comp5Vals, label = 'Component 5', ax=axis)


    plt.xticks(proteinList, proteinList, rotation='vertical')

    plt.legend(loc=2, prop={'size': 20})

