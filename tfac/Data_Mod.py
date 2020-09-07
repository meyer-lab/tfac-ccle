"""Data pre-processing and tensor formation"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorly.metrics.regression import variance as tl_var
from .dataHelpers import importLINCSprotein, ohsu_data
from .tensor import OHSU_parafac2_decomp, R2Xparafac2


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


def LINCSCleanUp():
    """Cleaning up LINCS data for PARAFAC2 column order"""
    LINCSprotein = importLINCSprotein()
    ind1 = LINCSprotein.loc[LINCSprotein['Time'] >= 24]
    ind2 = LINCSprotein.loc[LINCSprotein['Time'] == 0]
    ind = pd.concat([ind2, ind1])
    ind = ind.drop(columns='File')
    x = ['02_', '03_', '04_']
    y = ['0', '24', '48']
    for a in range(0, 3):
        for b in range(0, 3):
            ind = ind.replace(x[a] + 'RPPA_null1_' + '0' + y[b], 'cntrl' + y[b])
            ind = ind.replace(x[a] + 'RPPA_BMP2_' + y[b], 'BMP2_' + y[b])
            ind = ind.replace(x[a] + 'RPPA_EGF_' + y[b], 'EGF_' + y[b])
            ind = ind.replace(x[a] + 'RPPA_HGF_' + y[b], 'HGF_' + y[b])
            ind = ind.replace(x[a] + 'RPPA_IFNg_' + y[b], 'IFNg_' + y[b])
            ind = ind.replace(x[a] + 'RPPA_OSM_' + y[b], 'OSM_' + y[b])
            ind = ind.replace(x[a] + 'RPPA_TGFb_' + y[b], 'TGFb_' + y[b])
            ind = ind.replace(x[a] + 'RPPA_pbs_' + y[b], 'PBS_' + y[b])
    ind = ind.drop(columns=['Treatment', 'Time'])
    ind = ind.groupby(['Sample description']).mean()
    ind = ind.sort_values('Sample description')
    indT = ind.T
    treatmentsTime = indT.columns.tolist()
    proteins = indT.index.tolist()
    indT = indT.to_numpy()
    return indT, treatmentsTime, proteins


def dataCleanUp():
    """Cleaning up OHSU data for PARAFAC2 column order"""
    atac, cycIF, GCP, _, L1000, RNAseq, RPPA = ohsu_data()
    tr = ['BMP2_', 'EGF_', 'HGF_', 'IFNG_', 'OSM_', 'TGFB_', 'PBS_', 'ctrl_0']
    for r in range(0, 7):
        cycIF = cycIF.drop(columns=[tr[r] + '1', tr[r] + '4', tr[r] + '8'])
        GCP = GCP.drop(columns=[tr[r] + '4', tr[r] + '8'])
        L1000 = L1000.drop(columns=[tr[r] + '1', tr[r] + '4', tr[r] + '8'])
        RPPA = RPPA.drop(columns=[tr[r] + '1', tr[r] + '4', tr[r] + '8'])
    atac = atac.sort_index(axis=1)
    chromosomes = atac['peak'].to_list()
    atac = atac.drop(columns='peak').to_numpy()
    cycIF = cycIF.sort_index(axis=1)
    IFproteins = cycIF['feature'].to_list()
    cycIF = cycIF.drop(columns='feature').to_numpy()
    GCP = GCP.dropna()
    GCP = GCP.sort_index(axis=1)
    histones = GCP['histone'].to_list()
    GCP = GCP.drop(columns='histone').to_numpy()
    L1000 = L1000.sort_index(axis=1)
    geneExpression = L1000['probeset'].to_list()
    L1000 = L1000.drop(columns='probeset').to_numpy()
    RNAseq = RNAseq.sort_index(axis=1)
    RNAGenes = RNAseq['ensembl_gene_id'].tolist()
    RNAseq = RNAseq.drop(columns='ensembl_gene_id').to_numpy()
    RPPA = RPPA.sort_index(axis=1)
    RPPAProteins = RPPA['antibody'].tolist()
    RPPA = RPPA.drop(columns='antibody').to_numpy()
    return atac, cycIF, GCP, L1000, RNAseq, RPPA, chromosomes, IFproteins, histones, geneExpression, RNAGenes, RPPAProteins


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
    tensorSlices[1] = tensorSlices[1] * 3
    tensorSlices[2] = tensorSlices[2] * 20
    tensorSlices[3] = tensorSlices[3] * 12
    tensorSlices[5] = tensorSlices[5] * 20
    return tensorSlices

def R2X_OHSU(ax, p2slicesB):
    '''Creates R2X for OHSU PARAFAC2'''
    r2x = []
    compR2X = 10
    for i in range(1, compR2X + 1):
        parafac2tensor, error = OHSU_parafac2_decomp(p2slicesB, i)
        r2x.append(R2Xparafac2(p2slicesB, parafac2tensor))
    df = pd.DataFrame(r2x)
    comps = []
    for i in range(1, compR2X + 1):
        comps.append(i)
    df['Component'] = comps
    df.columns = ['Chromosomes', 'IFproteins', 'Histones', 'Gene Expression', 'RNA Genes', 'RPPA Proteins', 'Component']
    test = pd.melt(df, id_vars=['Component'])
    b = sns.scatterplot(data=test, x='Component', y='value', hue='variable', style='variable', ax=ax, s=100)
    b.set_xlabel("Component", fontsize=20)
    b.set_ylabel("R2X", fontsize=20)
    b.set_title("OHSU PARAFAC2")
    b.tick_params(labelsize=15)
    plt.legend(prop={'size': 15})
    ax.set_ylim(0, 1)


def OHSU_comp_plots(df, comps, ax):
    '''Plots treatments by each component from PARAFAC2'''
    sns.lineplot(data=df, x="Times", y=str(comps), hue="Treatments", ax=ax)
    ax.set_xlabel('Time (hr)')
    ax.set_title('Component ' + str(comps))


def proteinBoxPlot(ax, transformed, componentIn, sliceRows, title):
    '''Plots protein component in partial tucker factorization space with annotation of some outliers'''
    df = pd.DataFrame(transformed)
    prots = outliersForPlot(df, sliceRows)
    sns.boxplot(data=df, ax=ax)
    ax.set_xlabel("Component " + str(componentIn))
    ax.set_ylabel('Component Value')
    ax.set_title(title)
    for component in prots:
        offset_side = 0
        for outlier in prots[component]:
            if outlier[3]:
                if offset_side == 0:
                    ax.text(outlier[0] + .05, outlier[1] - .005, outlier[2], horizontalalignment='left', size='large', color='black', weight=100)
                    offset_side = 1
                elif offset_side == 1:
                    ax.text(outlier[0] - .05, outlier[1] - .005, outlier[2], horizontalalignment='right', size='large', color='black', weight=100)
                    offset_side = 0
            else:
                offset_side = 1 - offset_side


def outliersForPlot(transformed, sliceRows):
    '''Determines outliers based on IQR range by component and returns dictionary by component that allows annotation for OHSU PARAFAC2 analysis'''
    df = transformed.copy(deep=True)
    df['Proteins'] = sliceRows
    Q1 = df.quantile(.25)
    Q3 = df.quantile(.75)
    IQR = Q3 - Q1
    prots = {}
    for i in range(df.columns.size - 1):
        prots[i] = []
        for _, col in df.iterrows():
            if (col[i] < (Q1[i] - 1.7 * IQR[i])) or (col[i] > (Q3[i] + 1.7 * IQR[i])):
                tup = [i, col[i], col['Proteins'][:-1], True, True]
                prots[i].append(tup)
        prots[i].sort(key=lambda x: x[1])
        for idx, tup in enumerate(prots[i]):
            if idx < len(prots[i]) - 4:
                if tup[1] > prots[i][idx + 2][1] - .012 and tup[3] == tup[4]:
                    random1 = np.random.choice([0, 1, 1])
                    prots[i][idx + (random1 * 2)][3] = False
                    tup[4] = False
                    prots[i][idx + 2][4] = False
                elif tup[1] > prots[i][idx + 2][1] - .012 and tup[3]:
                    prots[i][idx + 2][3] = False
                    prots[i][idx + 2][4] = False
                if tup[1] > prots[i][idx + 4][1] - .012 and tup[3]:
                    random2 = np.random.randint(0, 2)
                    prots[i][idx + random2 * 4][3] = False
                    prots[i][idx + 4][4] = False
    return prots
