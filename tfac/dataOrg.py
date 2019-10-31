import numpy as np
import pandas as pd
from functools import reduce
from dataHelpers import importData

def extractData(filename, columns = None, row = 0, col = None):
    return pd.read_excel(filename, header = row, index_col = col, usecols = columns)

def findCommonGenes():
    '''
    Finds the set of unique gene names from the copy number, methylation, and gene expression dataset
    
    Returns:
            Numpy array of unique common gene names
    '''
    methylation, geneExp, copyNum = extractGeneNames()    
    commonGenes = reduce(np.intersect1d, (methylation, geneExp, copyNum))    
    return commonGenes

def filterData():
    '''
    Pushes the filtered data to synapse :D
    '''
    methFull = np.array(importData('NilayShah', 'nilayisthebest', 'Methylation All').values)
    copyFull = np.array(importData('NilayShah', 'nilayisthebest', 'Copy Number All').values)
    geneFull = np.array(importData('NilayShah', 'nilayisthebest', 'Gene Expression All').values)
    methylation, geneExp, copyNum = extractGeneNames()
    commmonGenes = findCommonGenes()
    
    methIndices = np.where(np.in1d(methylation, commonGenes))[0]
    copyIndices = np.where(np.in1d(copyNum, commonGenes))[0]
    geneIndices = np.where(np.in1d(geneExp, commonGenes))[0]
    
    methFiltered = methFull[methIndices, :]
    copyFiltered = copyFull[copyIndices, :]
    geneFiltered = geneFull[geneIndices, :]
    
    # Use synapse.store with file and activity functions to upload filtered data to synapse
    return
    
def extractGeneNames():
    ''' 
    Extracts sorted gene names from all data sets
    
    Returns:
            Order: Methylation, Gene Expression, Copy Number
            Returns three numpy arrays with gene names from aforementioned datasets
    '''
    data = extractData('data/GeneData_All.xlsx', 'A:C')
    data = data.to_numpy()
    
    methylation = np.append(data[:12158,0],data[12159:21338,0]).astype(str)
    geneExp = data[:,1].astype(str)
    copyNum = data[:23316,2].astype(str)
    
    return methylation, geneExp, copyNum


def extractCopy(dupes = False):
    ''' 
    Extracts out all duplicates data using excel file of gene names
    
    Returns:
            Order: Methylation, Gene Expression, Copy Number
            List of length 3 containing 3D arrays with 
            duplicate gene names, indices, and # of dupes corresponding to each name
            Also returns # of duplicates in each data set
    '''
    data = extractData('data/GeneData_All.xlsx', 'A:C')
    data = data.to_numpy()
    
    methylation = np.append(data[:12158,0],data[12159:21338,0])
    geneExp = data[:,1]
    copyNum = data[:23316,2]
    data = [methylation.astype(str),geneExp.astype(str),copyNum.astype(str)]
    
    if dupes:
        duplicates = np.zeros(3)
    
    returnVal = [] #creates list of 3 2D numpy arrays containing names and indices
    for i in range(len(data)):
        uData = np.unique(data[i], return_index = True, return_counts = True)
        
        if dupes:
            duplicates[i] = data[i].size - uData[0].size
            
        copyData = []
        idxData = []
        count = []
        for j in range(uData[0].size):
            if uData[2][j] != 1:
                copyData.append(uData[0][j])
                idxData.append(uData[1][j])
                count.append(uData[2][j])
        returnVal.append(np.array([copyData,idxData,count]))
        
    if dupes:
        return returnVal, duplicates
    else:
        return returnVal
        
#want to slice into array to create new datafile using indices stored