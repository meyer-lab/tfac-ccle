import numpy as np
import pandas as pd
from .dataHelpers import importLINCSprotein


def proteinNames():
    data=importLINCSprotein()
    data=data.drop(columns=['Treatment', 'Sample description', 'File', 'Time'],axis=1)
    proteinN=data.columns.values.tolist()
    return proteinN 
    
def componentWeights(decomps, proteinLabels):
    '''takes in array of protein decomp'd components and list of proteinNames and picks out top 3 protein weights
    components are by columns, proteins by rows'''
    i=np.shape(decomps)
    proteins=decomps[i[0]-1]
    proteinNum,compNum=np.shape(proteins)
    p_weights=np.zeros((3,compNum)) #'''stores indicies of 3 largest protein weights'''
    p_labels=[] #'''corresponding labels'''

    p_sorted=np.argsort(proteins,axis=0) #'''indirectly sorts by component from smallest to largest'''
    
    for y in range(0, compNum):
        for x in range(0, 3):
            p_weights[x,y] = p_sorted[proteinNum-(x+1),y]

        p_labels.append(topThreeProteins(proteins[:,y]))
        
    return p_weights, p_labels

def topThreeProteins(comp):
    """Returns the names of the top three weighted proteins for an individual protein component"""
    names = []
    for i in range(0, 3):
        temp = np.argmax(comp)
        names.append(proteinNames()[temp])
        comp[temp] = -100000
    return names
