"""
This creates Figure 2. This figure includes the Partial Tucker Treatment/Component Heat Map.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from ..dataHelpers import proteinNames
from ..tensor import decomp_to_flipped_factors

def makeCompPlots(axis1, axis2, axis3):
    """ Creates the component 1,4,5 plots. Component values versus time. """
    components = 5
    results, treatmentList, times = decomp_to_flipped_factors(components)
    treatmentTime = results[0]

    componeDict = {}

    #Initialize an empty dictionary of lists
    for treatment in treatmentList:
        componeDict[treatment] = []


    counter = 0
    #Create a dictionary with treatments as keys and the treatment value list as the value
    for keyTreatment in componeDict:
        counter1 = 0
        for timeList in treatmentTime[counter]:
            componeDict[keyTreatment].append(timeList[0])
            counter1+=1
        counter+=1

    counter=0

    #Plot the times against
    for treatment in treatmentList:
        sns.lineplot(x=times, y=componeDict[treatment], label=treatmentList[counter], ax=axis1, legend=True)
        counter+=1
    axis1.set_title("Component 1 v.s. Times")
    axis1.set(xlabel='Times', ylabel='Component Values')

    components = 5
    results, treatmentList, times = decomp_to_flipped_factors(components)
    treatmentTime = results[0]

    compfourDict = {}

    #Initialize an empty dictionary of lists
    for treatment in treatmentList:
        compfourDict[treatment] = []


    counter = 0
    #Create a dictionary with treatments as keys and the treatment value list as the value
    for keyTreatment in compfourDict:
        counter1 = 0
        for timeList in treatmentTime[counter]:
            compfourDict[keyTreatment].append(timeList[3])
            counter1+=1
        counter+=1

    counter = 0

    #Plot the times against
    for treatment in treatmentList:
        sns.lineplot(x=times, y=compfourDict[treatment], label=treatmentList[counter], ax=axis2, legend=True)
        counter+=1
    axis2.set_title("Component 4 v.s. Times")
    axis2.set(xlabel='Times', ylabel='Component Values')

    components = 5
    results, treatmentList, times = decomp_to_flipped_factors(components)
    treatmentTime = results[0]

    compfiveDict = {}

    #Initialize an empty dictionary of lists
    for treatment in treatmentList:
        compfiveDict[treatment] = []


    counter = 0
    #Create a dictionary with treatments as keys and the treatment value list as the value
    for keyTreatment in compfiveDict:
        counter1 = 0
        for timeList in treatmentTime[counter]:
            compfiveDict[keyTreatment].append(timeList[4])
            counter1+=1
        counter+=1

    counter = 0

    #Plot the times against
    for treatment in treatmentList:
        sns.lineplot(x=times, y=compfiveDict[treatment], label=treatmentList[counter], ax=axis3, legend=True)
        counter+=1
    axis3.set_title("Component 5 v.s. Times")
    axis3.set(xlabel='Times', ylabel='Component Values')

def makeHeatMap(axis):
    """ Create a heatmap of the variety of factors """
    result, treatments, times = decomp_to_flipped_factors(5)

    # this takes the list of arrays from results 1, of shape 1, 295, 5 and puts it into a dataFrame of shape 295, 5 (row, col)
    protMap = pd.DataFrame(result[1][0].T, columns=proteinNames())
    sel = np.max(np.absolute(protMap.values), axis=0)  # creates a np array of the max values of each of the proteins

    # This is the code to remove unnecessary proteins and label everything
    protMap = protMap.loc[:, sel > 0.2]

    # now protMap has all of the correct names and dropped proteins // transverse to make plot look better
    sns.heatmap(protMap.T, cmap='PiYG', vmin=-0.5, vmax=0.5, xticklabels=protMap.index + 1, ax=axis)
    axis.set_title("Protein Heatmap")
    axis.set(xlabel='Component', ylabel='Proteins')


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    row = 2
    col = 2
    ax, f = getSetup((8, 4), (row, col))

    # Add subplot labels
    subplotLabel(ax)

    makeCompPlots(ax[0], ax[1], ax[2])
    makeHeatMap(ax[3])

    fig, ax = plt.subplots(2, 2)
    return f
