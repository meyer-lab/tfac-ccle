import numpy as np
import pandas as pd
from functools import reduce
from dataHelpers import importData

def extractData(filename, columns = None, row = 0, col = None):
    return pd.read_excel(filename, header = row, index_col = col, usecols = columns)

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

def extractCellLines():
    ''' 
    Extracts sorted cell lines from all data sets
    
    Returns:
            Order: Methylation, Gene Expression, Copy Number
            Returns three numpy arrays with cell lines from aforementioned datasets
    '''
    data = extractData('data/CellLines_All.xlsx', 'A:C')
    data = data.to_numpy()
    
    methylation = data[:843,0].astype(str)
    geneExp = data[:1019,1].astype(str)
    copyNum = data[:,2].astype(str)
    
    return methylation, geneExp, copyNum

def findCommonGenes():
    '''
    Finds the set of unique gene names from the copy number, methylation, and gene expression dataset
    
    Returns:
            Numpy array of unique common gene names
    '''
    methylation, geneExp, copyNum = extractGeneNames()    
    commonGenes = reduce(np.intersect1d, (methylation, geneExp, copyNum))    
    return commonGenes

def findCommonCellLines():
    '''
    Finds the set of unique cell lines from the copy number, methylation, and gene expression dataset
    
    Returns:
            Numpy array of unique common cell lines
    '''
    methylation, geneExp, copyNum = extractCellLines()    
    commonCellLines = reduce(np.intersect1d, (methylation, geneExp, copyNum))    
    return commonCellLines

def filterData():
    '''
    Pushes the filtered data to synapse :D
    '''
    methData = importData('micah.bryant', '369DamnSheFine', 'Methylation All')
    geneData = importData('micah.bryant', '369DamnSheFine', 'Gene Expression All')
    copyData = importData('NilayShah', 'nilayisthebest', 'Copy Number All')
    
    methValues = np.array(methData.values)
    geneValues = np.array(geneData.values)
    copyValues = np.array(copyData.values)
    
    methIdx = np.array(methData.index)
    geneIdx = np.array(geneData.index)
    copyIdx = np.array(copyData.index)
    
    methG, geneG, copyG = extractGeneNames()
    methCL, geneCL, copyCL = extractCellLines()
    commonG = findCommonGenes()
    commonCL = findCommonCellLines()
    
    # Find indices of common genes in full dataset
    methGIndices = np.where(np.in1d(methG, commonG))[0]
    geneGIndices = np.where(np.in1d(geneG, commonG))[0]
    copyGIndices = np.where(np.in1d(copyG, commonG))[0]
    
    # Find indices of common cell lines in full dataset
    methCLIndices = np.where(np.in1d(methCL, commonCL))[0]
    geneCLIndices = np.where(np.in1d(geneCL, commonCL))[0]
    copyCLIndices = np.where(np.in1d(copyCL, commonCL))[0]
    
    
    methFiltered = methValues[methGIndices, methCLIndices]
    geneFiltered = geneValues[geneGIndices, geneCLIndices]
    copyFiltered = copyValues[copyGIndices, copyCLIndices]
    
    methDF = pd.DataFrame(data = methFiltered, index = methIdx[methGIndices], columns = commonCL)
    geneDF = pd.DataFrame(data = geneFiltered, index = geneIdx[geneGIndices], columns = commonCL)
    copyDF = pd.DataFrame(data = copyFiltered, index = copyIdx[copyGIndices], columns = commonCL)
    
    
       
    # Use synapse.store with file and activity functions to upload filtered data to synapse
    

def extractCopy(dupes = False, cellLines = False):
    ''' 
    Extracts out all duplicates data using excel file of gene names
    
    Returns:
            Order: Methylation, Gene Expression, Copy Number
            List of length 3 containing 3D arrays with 
            duplicate gene names, indices, and # of dupes corresponding to each name
            Also returns # of duplicates in each data set
    '''
    if cellLines:
        methylation, geneExp, copyNum = extractCellLines()
    else:
        methylation, geneExp, copyNum = extractGeneNames()
        
    data = [methylation,geneExp,copyNum]
    
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
