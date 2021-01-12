import numpy as np
import pandas as pd

def get_Flattened_Matrices(result, geneexpression, treatment_list, times):
    ids = geneexpression['ensembl_gene_id']
    geneexpression.drop('ensembl_gene_id', inplace = True, axis = 1)
    #create a 5x42 DataFrame of decompsed component values
    toflatten = result[0]
    flattened = []

    for treatment in range(7):
        flattened.append(pd.DataFrame(toflatten[treatment]).T)

    df = pd.concat(flattened, axis=1)
    
    #component values for control at T=0
    ctrl = df.iloc[:, 0]

    #Remove all times that aren't 24 or 48
    df.drop([0,1,2,3], inplace = True, axis = 1)
    
    #add control back in
    ctrl = pd.DataFrame(ctrl)
    df = pd.concat([ctrl,df], axis = 1)

    #rename columns
    cols = [str(x) + "_" + str(y) for x in treatment_list for y in times[-2:]]
    cols = ["ctrl_0"] + cols
    df.columns = cols

    #sort by column name
    df = df.reindex(sorted(df.columns), axis=1)
    geneexpression = geneexpression.reindex(sorted(geneexpression.columns), axis=1)

    #make dataframes into numpy arrays
    df = df.to_numpy()
    genexpression = geneexpression.to_numpy()
    return df, genexpression

def get_reconstruct(P,X):
    Ppinv = np.linalg.pinv(P.T)
    return Ppinv, np.matmul(Ppinv,X.T)

def find_reconstruction_norm(result, geneexpression, treatment_list, times):
    P, X = get_Flattened_Matrices(result, geneexpression, treatment_list, times)
    Ppinv, W = get_reconstruct(P,X)
    return np.linalg.norm(X.T-np.matmul(P.T,W)), X, P, W
