"""
This creates Figure 3.
"""

from .figureCommon import getSetup, subplotLabel
import numpy as np
import pandas as pd
from tensorly.decomposition import  partial_tucker,parafac2
import tensorly as tl
from tensorly.parafac2_tensor import parafac2_to_slice, apply_parafac2_projections
from ..Data_Mod import form_tensor
from ..tensor import partial_tucker_decomp, find_R2X_partialtucker, flip_factors
from ...pseudoinvnorm import get_Flattned_Matrices, get_reconstruct
from pybiomart import Server
import gseapy as gp
import copy
import matplotlib.pyplot as plt 
tl.set_backend("numpy")



def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    row = 3
    col = 1
    ax, f = getSetup((50, 50), (row, col))

    subplotLabel(ax)

    filtered=[]
    setty=[]
    for i in range(30):
        setty.append(pd.read_pickle('gsea_dfs/'+str(i)+'.pkl'))
        if ((i+1)%5==0):
            filtered.append(setty)
            setty = []
    filtered_dicts=[]
    for j in filtered:
        setty = []
        for i in j:
            my_dict = {}
            for r in range(len(i)):
                genes = i['genes'][r].split(';')
                for gene in genes:
                    if gene in my_dict:
                        my_dict[gene] +=1
                    else:
                        my_dict[gene] = 1
            setty.append(my_dict)
        filtered_dicts.append(setty)

    ###BELOW CODE PRINTS OUT MOST OR LEAST ENRICHED GENE SETS FOR A SPECIFIC COMPONENT

    #all enriched gene sets for component 5
    tot5 = pd.DataFrame()
    for i in range(6):
        tot5 = pd.concat([tot5, filtered[i][4]])

    #all enriched gene sets for component 4
    tot4 = pd.DataFrame()
    for i in range(6):
        tot4 = pd.concat([tot4, filtered[i][3]])

    #all enriched gene sets for component 3
    tot3 = pd.DataFrame()
    for i in range(6):
        tot3 = pd.concat([tot3, filtered[i][2]])

    #all enriched gene sets for component 2
    tot2 = pd.DataFrame()
    for i in range(6):
        tot2 = pd.concat([tot2, filtered[i][1]])

    #all enriched gene sets for component 1
    tot1 = pd.DataFrame()
    for i in range(6):
        tot1 = pd.concat([tot1, filtered[i][0]])

    #all enriched gene sets
    tot = pd.DataFrame()
    for i in range(6):
        for j in range(5):
            tot = pd.concat([tot, filtered[i][j]])
    just_score = ['es']
    size_then_score = ['geneset_size', 'es']
    matched_size_then_score = ['matched_size', 'es']

    all_tots = [tot1, tot2, tot3, tot4, tot5]

    heat1 = tot1.copy().reset_index()
    heat2 = tot2.copy().reset_index()
    heat3 = tot3.copy().reset_index()
    heat4 = tot4.copy().reset_index()
    heat5 = tot5.copy().reset_index()

    heat1 = heat1[["Term", "es"]]
    heat2 = heat2[["Term", "es"]]
    heat3 = heat3[["Term", "es"]]
    heat4 = heat4[["Term", "es"]]
    heat5 = heat5[["Term", "es"]]

    all_tots = [heat1, heat2, heat3, heat4, heat5]


    for m_df in all_tots:
        for row,cols in m_df.iterrows():
            if cols["Term"] not in list(heat1["Term"]):
                heat1 = heat1.append({"Term":cols["Term"], "es":0}, ignore_index = True)
            if cols["Term"] not in list(heat2["Term"]):
                heat2 = heat2.append({"Term":cols["Term"], "es":0}, ignore_index = True)
            if cols["Term"] not in list(heat3["Term"]):
                heat3 = heat3.append({"Term":cols["Term"], "es":0}, ignore_index = True)
            if cols["Term"] not in list(heat4["Term"]):
                heat4 = heat4.append({"Term":cols["Term"], "es":0}, ignore_index = True)
            if cols["Term"] not in list(heat5["Term"]):
                heat5 = heat5.append({"Term":cols["Term"], "es":0}, ignore_index = True)

    heat1.drop_duplicates(subset=['Term'], inplace = True)
    heat2.drop_duplicates(subset=['Term'], inplace = True)
    heat3.drop_duplicates(subset=['Term'], inplace = True)
    heat4.drop_duplicates(subset=['Term'], inplace = True)
    heat5.drop_duplicates(subset=['Term'], inplace = True)

    heat1.sort_values(by=['Term'], inplace = True)
    heat2.sort_values(by=['Term'], inplace = True)
    heat3.sort_values(by=['Term'], inplace = True)
    heat4.sort_values(by=['Term'], inplace = True)
    heat5.sort_values(by=['Term'], inplace = True)

    ax[0][1].scatter(heat4["es"], heat5['es'])
    ax[0][1].xlabel("Component 4 ES") 
    ax[0][1].ylabel("Component 5 ES") 
    ax[0][1].show()

    ax[0][2].scatter(heat1["Term"], heat1['es'],c='red', label = 'C1')
    ax[0][2].scatter(heat1["Term"], heat2['es'],c='blue', label = 'C2')
    ax[0][2].scatter(heat1["Term"], heat3['es'],c='green', label = 'C3')
    ax[0][2].scatter(heat1["Term"], heat4['es'],c='orange', label = 'C4')
    ax[0][2].scatter(heat1["Term"], heat5['es'],c='purple', label = 'C5')
    ax[0][2].legend()
    ax[0][2].xlabel("Set Name") 
    ax[0][2].ylabel("ES") 
    ax[0][2].xticks(rotation=90)
    ax[0][2].show()

    filtered=[]
    t_set=[]
    for i in range(12):
        t_set.append(pd.read_pickle('fold_dfs/'+str(i)+'.pkl'))
        if ((i+1)%2==0):
            filtered.append(t_set)
            t_set = []

    #all enriched gene sets for component 2
    tot2 = pd.DataFrame()
    for i in range(6):
        tot2 = pd.concat([tot2, filtered[i][1]])

    #all enriched gene sets for component 1
    tot1 = pd.DataFrame()
    for i in range(6):
        tot1 = pd.concat([tot1, filtered[i][0]])

    # #all enriched gene sets
    # tot = pd.DataFrame()
    # for i in range(6):
    #     for j in range(5):
    #         tot = pd.concat([tot, filtered[i][j]])




    heat1 = tot1.copy().reset_index()
    heat2 = tot2.copy().reset_index()


    heat1 = heat1[["Term", "es"]]
    heat2 = heat2[["Term", "es"]]


    all_tots = [heat1, heat2]


    for m_df in all_tots:
        for row,cols in m_df.iterrows():
            if cols["Term"] not in list(heat1["Term"]):
                heat1 = heat1.append({"Term":cols["Term"], "es":0}, ignore_index = True)
            if cols["Term"] not in list(heat2["Term"]):
                heat2 = heat2.append({"Term":cols["Term"], "es":0}, ignore_index = True)

    heat1.drop_duplicates(subset=['Term'], inplace = True)
    heat2.drop_duplicates(subset=['Term'], inplace = True)


    heat1.sort_values(by=['Term'], inplace = True)
    heat2.sort_values(by=['Term'], inplace = True)  

    ax[0][2].scatter(heat1["es"], heat2['es'])
    ax[0][2].xlabel("OSM ES") 
    ax[0][2].ylabel("IFN ES") 
    ax[0][2].show() 

    return f
