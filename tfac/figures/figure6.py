"""This creates Figure 6. Bar plot for percent variance explained by each gene expression component.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from tensorly.decomposition import  partial_tucker,parafac2
import tensorly as tl
from tensorly.parafac2_tensor import parafac2_to_slice, apply_parafac2_projections
from ..dataHelpers import ohsu_data
from ..tensor import decomp_to_flipped_factors, find_R2X_partialtucker
from tensorly.metrics.regression import variance as tl_var
from .figureCommon import subplotLabel, getSetup
tl.set_backend("numpy")

def get_Flattened_Matrices(result, geneexpression, treatment_list, times):
    '''Flattens treatment list and times dimensions into one treatment-time axis.
    Creates a new gene expression matrix (treatment-time x genes)'''
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

def find_gene_factors(result, geneexpression, treatment_list, times):
    '''Calculates pseudoinverse of flattened matrix, gene expression factors, flattened matrix, and gene expression matrix.'''
    P, X = get_Flattened_Matrices(result, geneexpression, treatment_list, times)
    Ppinv = np.linalg.pinv(P.T)
    W = np.matmul(Ppinv, X.T)
    return P, X, Ppinv, W

def var_diff(axis):
    '''Calculates amount of variance each variance explains from each component of gene expression factors.'''
    result, treatment_list, times = decomp_to_flipped_factors(10)
    _, _, _, _, _, RNAseq, _ = ohsu_data()
    P, X, Ppinv, W  = find_gene_factors(result, RNAseq, treatment_list, times)
    residuals = np.zeros(5)
    for i in range(1, 6):
        #removes respective gene expression and treatment-time per iteration
        removeGene = np.delete(W, i-1, 0) 
        removeTT = np.delete(P, i-1, 0)
        #reconstructs factors with removed row/col combo
        gene_reconst = np.matmul(removeGene.T, removeTT)
        #calculates the percent variance between remove-one factors
        residuals[i-1] = tl_var(gene_reconst - (np.matmul(W.T, P)))/tl_var(RNAseq.to_numpy())
    sns.barplot(np.arange(len(residuals)), residuals, ax = axis)
    axis.set_xlabel("Component Removed")
    axis.set_ylabel("Difference in Percent Variance")
    axis.set_xticklabels(['1', '2', '3', '4', '5'])

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    row = 1
    col = 1
    ax, f = getSetup((4, 4), (row, col))
    var_diff(ax[0])
    subplotLabel(ax)
    return f