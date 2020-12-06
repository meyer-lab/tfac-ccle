"""Data pre-processing and tensor formation"""
import pandas as pd
import numpy as np
from tensorly.metrics.regression import variance as tl_var
from .dataHelpers import importLINCSprotein, dataCleanUp


def data_mod(x, df=None):
    """Creates a slice of the data tensor corresponding to the inputted treatment"""
    if not isinstance(df, pd.core.frame.DataFrame):
        df = importLINCSprotein()
    spec_df = df.loc[(df["Treatment"] == "Control") | (df["Treatment"] == x)]
    times = spec_df["Time"].to_numpy().tolist()
    spec_df = spec_df.drop(columns=["Sample description", "Treatment", "Time"])
    y = spec_df.to_numpy()
    return y, spec_df, times


def form_tensor():
    """Creates tensor in numpy array form and returns tensor, treatments, and time"""
    df = importLINCSprotein()
    tempindex = df["Sample description"]
    tempindex = tempindex[:36]
    i = 0
    for a in tempindex:
        tempindex[i] = a[3:]
        i += 1
    treatments = df["Treatment"][0:36]
    times = df["Time"][0:36]
    df = df.drop(["Sample description"], axis=1)
    by_row_index = df.groupby(df.index)
    df_means = by_row_index.mean()
    df_means.insert(0, "Treatment", value=treatments)
    df_means.insert(0, "Sample description", tempindex)
    unique_treatments = np.unique(df_means["Treatment"].values).tolist()
    unique_treatments.remove("Control")

    slices = []
    for treatment in unique_treatments:
        array, _, times = data_mod(treatment, df_means)
        slices.append(array)
    tensor = np.stack(slices)
    return tensor, unique_treatments, times

def form_parafac2_tensor():
    """Creates tensor in numpy form and returns tensor, treatment by time, LINCS proteins, ATAC chromosomes, IF proteins, GCP histones, L1000 gene expression, RNA gene sequence, and RPPA proteins"""
    atacM, cycIFM, GCPM, L1000M, RNAseqM, RPPAM, chromosomes, IFproteins, histones, geneExpression, RNAGenes, RPPAProteins = dataCleanUp()
    p2slices = [atacM, cycIFM, GCPM, L1000M, RNAseqM, RPPAM]
    for x in range(len(p2slices)):
        df = pd.DataFrame(p2slices[x], columns=['BMP2_24', 'BMP2_48', 'EGF_24', 'EGF_48', 'HGF_24', 'HGF_48', 'IFNG_24', 'IFNG_48', 'OSM_24', 'OSM_48', 'TGFB_24', 'TGFB_48', 'PBS_24', 'PBS_48', 'ctrl_0'])
        df = df.replace('5.!21302089907794', 5.21302089907794)
        df = df.replace('6>029875019867487', 6.029875019867487)
        df = df.replace('6>035425175013243', 6.035425175013243)
        df = df.replace('6.66!312194765656', 6.66312194765656).astype(float)
        df = df - (df.apply(np.mean)) / df.apply(np.std)
        df = (df.sub(df.apply(np.mean, axis=1).to_list(), axis=0)).div(df.apply(np.std, axis=1).to_list(), axis=0)
        p2slices[x] = df.to_numpy()
    return p2slices, chromosomes, IFproteins, histones, geneExpression, RNAGenes, RPPAProteins

def ohsu_var(tensorSlices):
    '''Rebalances variance of all tensor slices to 1'''
    for x, val in enumerate(tensorSlices):
        var = tl_var(tensorSlices[x])
        tensorSlices[x] = (tensorSlices[x]) / (var ** 0.5)
    #tensorSlices[1] = tensorSlices[1] * 3
    tensorSlices[2] = tensorSlices[2] * 20
    #tensorSlices[3] = tensorSlices[3] * 12
    tensorSlices[5] = tensorSlices[5] * 15
    return tensorSlices

