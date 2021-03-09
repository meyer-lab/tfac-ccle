"""
This creates Figure 2. This figure includes the Partial Tucker Treatment/Component Heat Map.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from figureCommon import subplotLabel, getSetup
from ..tensor import partial_tucker_decomp, flip_factors, decomp_to_flipped_factors
from ..Data_Mod import form_tensor
from ..dataHelpers import importLINCSprotein


def createCompPlots():
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

    #Plot the times against 
    for treatment in treatmentList:
        plt.plot(times, componeDict[treatment], label=treatmentList[counter])
        counter+=1
    plt.legend()
    plt.title("Component 1 v.s. Times")

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
        plt.plot(times, compfourDict[treatment], label=treatmentList[counter])
        counter+=1
    plt.legend()
    plt.title("Component 4 v.s. Times")

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
        plt.plot(times, compfiveDict[treatment], label=treatmentList[counter])
        counter+=1
    plt.legend()
    plt.title("Component 5 v.s. Times")

def createHeatMap():
    result = partial_tucker_decomp(form_tensor()[0], [2], rank=5)
    result = flip_factors(result)

    # this takes the list of arrays from results 1, of shape 1, 295, 5 and puts it into a dataFrame of shape 295, 5 (row, col)
    protMap = pd.DataFrame(result[1][0].T, columns=proteinNames())
    sel = np.max(np.absolute(protMap.values), axis=0)  # creates a np array of the max values of each of the proteins

    # This is the code to remove unnecessary proteins and label everything
    protMap = protMap.loc[:, sel > 0.2]

    # now protMap has all of the correct names and dropped proteins // transverse to make plot look better
    sns.heatmap(protMap.T, cmap='PiYG', vmin=-0.5, vmax=0.5, xticklabels=protMap.index + 1)

def proteinFactorsDist():
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

    plt.scatter(proteinList, comp1Vals, label = 'Component 1')
    plt.scatter(proteinList, comp2Vals, label = 'Component 2')
    plt.scatter(proteinList, comp3Vals, label = 'Component 3')
    plt.scatter(proteinList, comp4Vals, label = 'Component 4')
    plt.scatter(proteinList, comp5Vals, label = 'Component 5')


    plt.xticks(proteinList, proteinList, rotation='vertical')

    plt.legend(loc=2, prop={'size': 20})

def makeFigure():
    """ Get a list of the axis objects and create a figure. """

    createCompPlots()
    createHeatMap()
    proteinFactorsDist()

    component = 5
    tensor, treatment_list, times = form_tensor()
    pre_flip_result = partial_tucker_decomp(tensor, [2], component)

    result = flip_factors(pre_flip_result)

    compList = ["1", "2", "3", "4", "5"]
    trmtTimeList = []
    for i, name in enumerate(treatment_list):
        for t in range(len(times)):
            trmtTimeList.append(name + "_" + str(times[t]))
    trmtTime = np.reshape(trmtTimeList, (7, 6))
    trmtMap = pd.DataFrame()
    for y in range(len(trmtTime)):
        temp = pd.DataFrame(data=result[0][y], index=trmtTime[y], columns=compList)
        trmtMap = trmtMap.append(temp)
        palette = sns.cubehelix_palette(len(trmtTime), light=0.8, dark=0.2, reverse=True, start=1, rot=-2)
    dict = {}
    time_tick = []
    for x in range(7):
        for y in range(6):
            dict[str(trmtTime[x, y])] = palette[x]
            time_tick.append(times[y])
    colors = pd.Series(trmtTimeList, index=trmtMap.index).map(dict)
    f = sns.clustermap(trmtMap, row_cluster=False, col_cluster=False, row_colors=colors, cmap="PiYG", center=0, yticklabels=time_tick)
    for val, label in enumerate(trmtTime[:, 0]):
        f.ax_col_dendrogram.bar(0, 0, color=dict[label], label=treatment_list[val], linewidth=0)
    f.ax_col_dendrogram.legend(loc="center left", ncol=5)
    f.fig.suptitle("Treatment-Time vs. Component")
    return f
