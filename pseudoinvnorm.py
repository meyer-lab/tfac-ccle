import numpy as np
import pandas as pd
from tensorly.decomposition import  partial_tucker,parafac2
import tensorly as tl
from tensorly.parafac2_tensor import parafac2_to_slice, apply_parafac2_projections
from tfac.Data_Mod import form_parafac2_tensor, ohsu_var, form_tensor
from tfac.tensor import partial_tucker_decomp, find_R2X_partialtucker, flip_factors
tl.set_backend("numpy")

def get_Flattened_Matrices():
    """return the patient matrix and the gene expression matrix as numpy arrays"""
    #decompse the tensor
    component = 5
    tensor, treatment_list, times = form_tensor()
    pre_flip_result = partial_tucker_decomp(tensor, [2], component)
    result = flip_factors(pre_flip_result)
    #load the gene expression data
    geneexpression = pd.read_csv("tfac/data/ohsu/MDD_RNAseq_Level4.csv")
    ids = geneexpression["ensembl_gene_id"]
    geneexpression.drop("ensembl_gene_id", inplace = True, axis = 1)
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
    """Return the pseudo inverse to the patient matrix as well as the pseudo inv multiplied by the gene expression matrix"""
    Ppinv = np.linalg.pinv(P.T)
    return Ppinv, np.matmul(Ppinv,X.T)

def find_reconstruction_norm():
    """return the norm of the difference in original and reconstructed gene expression matrices"""
    P, X = get_Flattened_Matrices()
    Ppinv, W = get_reconstruct(P,X)
    return np.linalg.norm(X.T-np.matmul(P.T,W))
