"""
This creates Figure 3.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from ..tensor import partial_tucker_decomp, flip_factors
from ..Data_Mod import form_tensor
from ..dataHelpers import importLINCSprotein, proteinNames

def protein_heatmap():
    component = 5
    tensor, treatment_list, times = form_tensor()
    pre_flip_result = partial_tucker_decomp(tensor, [2], component)

    result = flip_factors(pre_flip_result)
    #This is the code to make the protein dataframe
    protMap = pd.DataFrame()
    #this takes the list of arrays from results 1, of shape 1, 295, 5 and puts it into a dataFrame of shape 295, 5 (row, col)
    temp = pd.DataFrame(np.concatenate(result[1]))
    protMap = temp.T #I just make the transpose so that we have each of the 5 rows as a component, and all 295 col as proteins
    pNames = proteinNames() #get a list of our protein names

    #This is the code to remove unnecessary proteins and label everything
    n = 0 #used to search protMap
    ind = 0 #keeps track of column number
    size = 295
    #loop through the entire dataframe
    while(n < size):
        #check individually to see if all of the weights are insignificant, and if so, remove the whole protein
        if (abs(protMap.iat[0, n]) <= 0.2 and abs(protMap.iat[1,n]) <= 0.2 and abs(protMap.iat[2,n]) <= 0.2 and abs(protMap.iat[3,n]) <= 0.2 and abs(protMap.iat[4,n]) <= 0.2):
            #now we want to remove that column, reduce size, and keep moving forward
            protMap = protMap.drop(ind, axis = 1) #want to use the index to drop because of dataframe column title is ind
            ind += 1 #move index forward too
            size -= 1
        else: #otherwise we have something significant in at least one of the components!
            protMap = protMap.rename(columns={ind : pNames[ind]}) #rename the column title, which is ind to the name from pNames
            n+=1 #now we can look at the next one and don't have to stay stagnant
            ind += 1
    #now protMap has all of the correct names and dropped proteins // transverse to make plot look better
    protMap1 = protMap.T
    sns.heatmap(protMap1, cmap = 'PiYG', vmin = -0.5, vmax = 0.5, xticklabels = protMap.index + 1, ax = ax[0])


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    row = 1
    col = 1
    ax, f = getSetup((7, 12), (row, col))
    ax[0].set_title("Protein-Component Heat Plot")
    protein_heatmap()
    subplotLabel(ax)
    return f