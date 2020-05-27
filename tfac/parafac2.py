import pandas as pd
import numpy as np

tensor, cytokines, GeneIDs = form_MRSA_tensor()
print(tensor)

def form_MRSA_tensor():
    dfClin, dfCoh = importClinicalMRSA()
    dfCyto = clinicalCyto(dfClin, dfCoh)
    dfCyto = dfCyto.sort_values(by='sid')
    dfCyto = dfCyto.set_index('sid')
    cytokines = dfCyto.columns
    
    dfExp = importExpressionData()
    dfExp = dfExp.T
    geneIDs = dfExp.iloc[0, 0:].to_list()
    dfExp.columns = geneIDs
    dfExp = dfExp.drop('Geneid')
    
    cytoNumpy = dfCyto.to_numpy().T
    expNumpy = dfExp.to_numpy().T
    
    tensor_slices = [cytoNumpy, expNumpy]
    
    return tensor_slices, cytokines, geneIDs

def importClinicalMRSA():
    """import clincal MRSA data"""
    dataClin = pd.read_csv(join(path_here, "tfac/data/mrsa/mrsa_s1s2_clin+cyto_073018.csv"))
    dataCohort = pd.read_csv(join(path_here, "tfac/data/mrsa/clinical_metadata_cohort1.txt"), delimiter='\t')
    return dataClin, dataCohort

def clinicalCyto(dataClinical, dataCohort):
    """isolate cytokine data from clinical"""
    rowSize, colSize = dataClinical.shape
    patientID = list(dataClinical["sid"])
    
    dataClinical = dataClinical.drop(dataClinical.iloc[:, 0:3], axis=1)
    dataClinical = dataClinical.drop(dataClinical.iloc[:, 1:207], axis=1)
    
    """isolate patient IDs from cohort 1"""
    dataCohort = dataCohort.drop(columns=['age','gender','race','sampletype','pair','outcome_txt'], axis=1)
    cohortID = list(dataCohort["sample"])
    IDSize, column = dataCohort.shape
   
    cytokineData = pd.DataFrame()
    
    for y in range(0, rowSize):
        for z in range(0, IDSize):
            if (cohortID[z]).find(str(patientID[y])) != -1:
                temp = dataClinical.loc[dataClinical['sid']==patientID[y]]
                cytokineData = pd.concat([temp, cytokineData])
    cytokineData.sort_values(by=['sid'])
    return cytokineData

def importExpressionData():
    """import expression data"""
    df = pd.read_table(join(path_here, "tfac/data/mrsa/expression_counts_cohort1.txt"))
    df.drop(["Chr", "Start", "End", "Strand", "Length"], inplace=True, axis=1)
    return df
