"""
This creates Figure 4. Gene expression R2X with flattened matrix dimension reconstruction.
"""

import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from ..CMTF import perform_CMTF
from ..dataHelpers import form_tensor, proteinNames


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((8, 12), (1, 1))

    # Add subplot labels
    subplotLabel(ax)

    makeProteinListDistribution(ax[0])

    return f


def makeProteinListDistribution(ax):
    """ Create the protein list distribution with components on the x axis, component values on the y axis and outlier proteins labelled """
    components = 5
    tensor, rTensor, _, _ = form_tensor()
    results = perform_CMTF(tensor, rTensor, r=components)

    proteinFactors = results[1][0]
    proteinList = proteinNames()

    comp1Vals = []
    comp2Vals = []
    comp3Vals = []
    comp4Vals = []
    comp5Vals = []

    comp1 = []
    comp2 = []
    comp3 = []
    comp4 = []
    comp5 = []

    proteinComp1 = []
    proteinComp2 = []
    proteinComp3 = []
    proteinComp4 = []
    proteinComp5 = []

    proteinCounter1 = 0
    proteinCounter2 = 0
    proteinCounter3 = 0
    proteinCounter4 = 0
    proteinCounter5 = 0

    for compVals in proteinFactors:
        if(compVals[0] < -.15 or compVals[0] > .15):
            comp1Vals.append(compVals[0])
            comp1.append("Component 1")
            proteinComp1.append(proteinList[proteinCounter1])
            proteinCounter1 += 1
        if(compVals[1] < -.15 or compVals[1] > .15):
            comp2Vals.append(compVals[1])
            comp2.append("Component 2")
            proteinComp2.append(proteinList[proteinCounter2])
            proteinCounter2 += 1
        if(compVals[2] < -.15 or compVals[2] > .15):
            comp3Vals.append(compVals[2])
            comp3.append("Component 3")
            proteinComp3.append(proteinList[proteinCounter3])
            proteinCounter3 += 1
        if(compVals[3] < -.15 or compVals[3] > .15):
            comp4Vals.append(compVals[3])
            comp4.append("Component 4")
            proteinComp4.append(proteinList[proteinCounter4])
            proteinCounter4 += 1
        if(compVals[4] < -.15 or compVals[4] > .15):
            comp5Vals.append(compVals[4])
            comp5.append("Component 5")
            proteinComp5.append(proteinList[proteinCounter5])
            proteinCounter5 += 1

    sns.scatterplot(x=comp1, y=comp1Vals, ax=ax, color='blue')
    sns.scatterplot(x=comp2, y=comp2Vals, ax=ax, color='orange')
    sns.scatterplot(x=comp3, y=comp3Vals, ax=ax, color='yellow')
    sns.scatterplot(x=comp4, y=comp4Vals, ax=ax, color='green')
    sns.scatterplot(x=comp5, y=comp5Vals, ax=ax, color='red')

    for line in range(0, len(comp1Vals)):
        if(line % 2 == 0):
            ax.text(comp1[line], comp1Vals[line], proteinList[line], horizontalalignment='right', size=6, color='black', weight='semibold')
        else:
            ax.text(comp1[line], comp1Vals[line], proteinList[line], horizontalalignment='left', size=6, color='black', weight='semibold')

    for line in range(0, len(comp2Vals)):
        if(line % 2 == 0):
            ax.text(comp2[line], comp2Vals[line], proteinList[line], horizontalalignment='right', size=6, color='black', weight='semibold')
        else:
            ax.text(comp2[line], comp2Vals[line], proteinList[line], horizontalalignment='left', size=6, color='black', weight='semibold')

    for line in range(0, len(comp3Vals)):
        if(line % 2 == 0):
            ax.text(comp3[line], comp3Vals[line], proteinList[line], horizontalalignment='right', size=6, color='black', weight='semibold')
        else:
            ax.text(comp3[line], comp3Vals[line], proteinList[line], horizontalalignment='left', size=6, color='black', weight='semibold')

    for line in range(0, len(comp4Vals)):
        if(line % 2 == 0):
            ax.text(comp4[line], comp4Vals[line], proteinList[line], horizontalalignment='right', size=6, color='black', weight='semibold')
        else:
            ax.text(comp4[line], comp4Vals[line], proteinList[line], horizontalalignment='left', size=6, color='black', weight='semibold')

    for line in range(0, len(comp5Vals)):
        if(line % 2 == 0):
            ax.text(comp5[line], comp5Vals[line], proteinList[line], horizontalalignment='right', size=6, color='black', weight='semibold')
        else:
            ax.text(comp5[line], comp5Vals[line], proteinList[line], horizontalalignment='left', size=6, color='black', weight='semibold')

    comp1Vals.clear()
    comp2Vals.clear()
    comp3Vals.clear()
    comp4Vals.clear()
    comp5Vals.clear()

    comp1.clear()
    comp2.clear()
    comp3.clear()
    comp4.clear()
    comp5.clear()

    for compVals in proteinFactors:
        if(compVals[0] > -.15 or compVals[0] < .15):
            comp1Vals.append(compVals[0])
            comp1.append("Component 1")
        if(compVals[1] > -.15 or compVals[1] < .15):
            comp2Vals.append(compVals[1])
            comp2.append("Component 2")
        if(compVals[2] > -.15 or compVals[2] < .15):
            comp3Vals.append(compVals[2])
            comp3.append("Component 3")
        if(compVals[3] > -.15 or compVals[3] < .15):
            comp4Vals.append(compVals[3])
            comp4.append("Component 4")
        if(compVals[4] > -.15 or compVals[4] < .15):
            comp5Vals.append(compVals[4])
            comp5.append("Component 5")

    sns.scatterplot(x=comp1, y=comp1Vals, ax=ax, color='blue', alpha=.05)
    sns.scatterplot(x=comp2, y=comp2Vals, ax=ax, color='orange', alpha=.05)
    sns.scatterplot(x=comp3, y=comp3Vals, ax=ax, color='yellow', alpha=.05)
    sns.scatterplot(x=comp4, y=comp4Vals, ax=ax, color='green', alpha=.05)
    sns.scatterplot(x=comp5, y=comp5Vals, ax=ax, color='red', alpha=.05)

    ax.set_title("Component Values for Proteins")
    ax.set(xlabel='Components', ylabel='Component Values')
