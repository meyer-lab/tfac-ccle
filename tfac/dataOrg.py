import numpy as np
import pandas as pd

def extractData(filename, columns = None, row = 0, col = None):
    return pd.read_excel(filename, header = row, index_col = col, usecols = columns)

def extractCopy(dupes = False):
    ''' 
    Extracts out all duplicates data using excel file of gene names
    
    Returns:
            Order: Methylation, Gene Expression, Copy Number
            List of length 3 containing 2D arrays with duplicate gene names and indices found
            Also returns # of duplicates in each data set
    '''
    data = extractData('data/GeneData_All.xlsx', 'A:C')
    data = data.to_numpy()
    
    methylation = np.append(data[:12158,0],data[12159:,0])
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
        for i in range(uData[0].size):
            if uData[2][i] != 1:
                copyData.append(uData[0][i])
                idxData.append(uData[1][i])
        returnVal.append(np.array([copyData,idxData]))
        
    if dupes:
        return returnVal, duplicates
    else:
        return returnVal
        