"""
This creates Figure 5. R2X for proteins with flattened dimension reconstruction.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from tensorly.metrics.regression import variance as tl_var
from ..Data_Mod import form_tensor
from ..tensor import partial_tucker_decomp, find_R2X_partialtucker, flip_factors
from ..pseudoinvnorm import get_reconstruct
from .figureCommon import getSetup, subplotLabel


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    row = 1
    col = 1
    ax, f = getSetup((4, 4), (row, col))
    protein_R2X(ax[0])
    subplotLabel(ax)
    return f


def get_Flattened_Matrices(result, data, treatment_list, times):
    # create a 5x42 DataFrame of decompsed component values for protein treatment-times
    toflatten = result[0]  # treatment/comp
    flattened = []

    for treatment in range(7):
        flattened.append(pd.DataFrame(toflatten[treatment]).T)

    df = pd.concat(flattened, axis=1)

    # component values for control at T=0
    ctrl = df.iloc[:, 0]

    # add control back in
    ctrl = pd.DataFrame(ctrl)
    df = pd.concat([ctrl, df], axis=1)

    # rename columns
    cols = [str(x) + "_" + str(y) for x in treatment_list for y in times]
    cols = ["ctrl_0"] + cols
    df.columns = cols

    # sort by column name
    df = df.reindex(sorted(df.columns), axis=1)
    data = data.reindex(sorted(data.columns), axis=1)

    # make dataframes into numpy arrays
    df = df.to_numpy()
    data = data.to_numpy()
    return df, data


def find_gene_factors(result, data, treatment_list, times):
    P, X = get_Flattened_Matrices(result, data, treatment_list, times)
    Ppinv, W = get_reconstruct(P, X)
    return P, X, Ppinv, W


def protein_R2X(ax):
    # data processing for protein
    tensor, treatment_list, times = form_tensor()
    R2X = np.zeros(13)
    protein = pd.read_csv('tfac/data/ohsu/MDD_RPPA_Level4.csv')
    protein = protein.drop('antibody', axis=1)
    colName = [
        'ctrl_0',
        'BMP2_0',
        'BMP2_1',
        'BMP2_4',
        'BMP2_8',
        'BMP2_24',
        'BMP2_48',
        'EGF_0',
        'EGF_1',
        'EGF_4',
        'EGF_8',
        'EGF_24',
        'EGF_48',
        'HGF_0',
        'HGF_1',
        'HGF_4',
        'HGF_8',
        'HGF_24',
        'HGF_48',
        'IFNg_0',
        'IFNg_1',
        'IFNg_4',
        'IFNg_8',
        'IFNg_24',
        'IFNg_48',
        'OSM_0',
        'OSM_1',
        'OSM_4',
        'OSM_8',
        'OSM_24',
        'OSM_48',
        'PBS_0',
        'PBS_1',
        'PBS_4',
        'PBS_8',
        'PBS_24',
        'PBS_48',
        'TGFb_0',
        'TGFb_1',
        'TGFb_4',
        'TGFb_8',
        'TGFb_24',
        'TGFb_48']
    ctrl = protein['ctrl_0']
    for x, i in enumerate(treatment_list):
        protein.insert(x, i + '_0', ctrl, allow_duplicates=True)
    for y in times:
        protein.rename(
            columns={
                'TGFB_1': 'TGFb_1',
                'TGFB_4': 'TGFb_4',
                'TGFB_8': 'TGFb_8',
                'TGFB_24': 'TGFb_24',
                'TGFB_48': 'TGFb_48',
                'IFNG_1': 'IFNg_1',
                'IFNG_4': 'IFNg_4',
                'IFNG_8': 'IFNg_8',
                'IFNG_24': 'IFNg_24',
                'IFNG_48': 'IFNg_48'},
            inplace=True)
    protein = protein.reindex(colName, axis=1)
    # R2X calculation
    for i in range(1, 13):
        pre_flip_result = partial_tucker_decomp(tensor, [2], i)
        result = flip_factors(pre_flip_result)
        P, X, Ppinv, W = find_gene_factors(result, protein, treatment_list, times)
        protein_redone = np.matmul(W.T, P)
        R2X[i] = 1 - tl_var(protein_redone - X) / tl_var(protein.to_numpy())
    sns.scatterplot(np.arange(len(R2X)), R2X, ax=ax)
    ax.set_xlabel("Rank Decomposition")
    ax.set_ylabel("R2X")
    ax.set_title("Variance Explained in Protein Data")
    ax.set_yticks([0, .2, .4, .6, .8, 1])
