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

def protein_heatmap(m_axis):
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
    maxValues = np.max(protMap, axis=0) #creates a np array of the max values of each of the proteins
    boolsOfVal = np.empty((1, 295), dtype = bool) #create a boolean array to store T/F
    np.logical_or(maxValues > 0.2, maxValues < -0.2, out = boolsOfVal) #check all of the values to see if they are significant (0.2) 

    #This is the code to remove unnecessary proteins and label everything
    n = 0 #used to search protMap
    size = 295
    #loop through the entire dataframe
    while(n < size):
        #check if we need to drop (false) or not (true)
        if (boolsOfVal[0][n] == False): #now we want to remove that column from protMap
            protMap = protMap.drop(n, axis = 1) #use n to drop because n is tracking in both boolsOfVal and protMap
        else: #otherwise we have something significant!
            protMap = protMap.rename(columns={n : pNames[n]}) #rename the column title, which is n to the name from pNames
        n += 1
    #now protMap has all of the correct names and dropped proteins // transverse to make plot look better
    protMap1 = protMap.T
    sns.heatmap(protMap1, cmap = 'PiYG', vmin = -0.5, vmax = 0.5, xticklabels = protMap.index + 1, ax = m_axis)


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    row = 1
    col = 1
    ax, f = getSetup((7, 12), (row, col))
    ax[0].set_title("Protein-Component Heat Plot")
    protein_heatmap(ax[0])
    subplotLabel(ax)
    return f