"""
This creates Figure 2 - protein factors.
(a) heatmap of the proteins in four subplots
(b) ?
(c-e) components vs time for various treatments.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.cluster.hierarchy as sch
from .common import subplotLabel, getSetup
from tensorpack import perform_CP, Decomposition
from ..dataHelpers import import_LINCS_CCLE, import_LINCS_MEMA, proteinNames


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((20, 40), (5, 3))

    # mema MCF10A
    MCF10A, ligand1, ecm1, meas1 = import_LINCS_MEMA("mcf10a_ssc_Level4.tsv.xz")
    # mema HMEC240L
    HMEC240, ligand2, ecm2, meas2 = import_LINCS_MEMA("hmec240l_ssc_Level4.tsv.xz")
    # mema HMEC122L
    HMEC122, ligand3, ecm3, meas3 = import_LINCS_MEMA("hmec122l_ssc_Level4.tsv.xz")

    g = generate_components_CCLE(ax)
    g1 = generate_components_MEMA(MCF10A, 6, ligand1, ecm1, meas1, ax, [6,7,8])
    g2 = generate_components_MEMA(HMEC240, 6, ligand2, ecm2, meas2, ax, [9,10,11])
    g3 = generate_components_MEMA(HMEC122, 6, ligand3, ecm3, meas3, ax, [12,13,14])

    # Add subplot labels
    subplotLabel(ax)

    return f

def generate_components_CCLE(ax):

    tensor, drugs, times = import_LINCS_CCLE()
    s = 4 ** 0
    protData = tensor[:,:,:295] * s
    rnaData = tensor[:,:,295:]
    tensor = np.append(protData, rnaData, axis=2)

    protein_names = proteinNames()
    RNAseq = pd.read_csv("tfac/data/ohsu/module_expression.csv", delimiter=",")
    gene_modules = list(RNAseq["Unnamed: 0"])

    ccle, drugs, times = import_LINCS_CCLE()
    tfac = Decomposition(ccle, max_rr=6)
    tfac.perform_tfac()
    tfac = tfac.tfac[-1]

    treat     = pd.DataFrame(tfac.factors[0],columns=[f"Cmp. {i}" for i in np.arange(1, tfac.rank + 1)], index=drugs)
    time_     = pd.DataFrame(tfac.factors[1],columns=[f"Cmp. {i}" for i in np.arange(1, tfac.rank + 1)], index=times)
    prots     = pd.DataFrame(tfac.factors[2][:295,:],columns=[f"Cmp. {i}" for i in np.arange(1, tfac.rank + 1)], index=protein_names)
    geneMods  = pd.DataFrame(tfac.factors[2][295:,:],columns=[f"Cmp. {i}" for i in np.arange(1, tfac.rank + 1)], index=gene_modules)

    g1 = sns.heatmap(treat, cmap="PiYG", center=0, yticklabels=True, vmin=-1, vmax=1, square=True, ax=ax[0])
    g1.set_yticklabels(g1.get_yticklabels(), rotation = 0)

    g2 = sns.heatmap(prots, cmap="PiYG", center=0, yticklabels=True, vmin=-1, vmax=1, ax=ax[1])
    g2_y = np.arange(0,len(protein_names),10)
    g2.set_yticks(g2_y)
    g2.set_yticklabels(g2_y, rotation = 0, fontsize=5)
    g2.set_ylabel('Protein Indices')
    g2.set_title('Protein Components')

    g3 = sns.heatmap(geneMods, cmap="PiYG", center=0, yticklabels=True, vmin=-1, vmax=1, ax=ax[2])
    g3_y = np.arange(0,len(geneMods),10)
    g3.set_yticks(g3_y)
    g3.set_yticklabels(g3_y, rotation = 0, fontsize=5)
    g3.set_ylabel('Gene Module Indices')
    g3.set_title('Gene Module Components')

    g4 = sns.lineplot(data=time_, palette='colorblind', dashes=False, ax=ax[3])
    g4.set(ylim=(-1, 1))
    g4.set_xlabel('Time (hrs)')
    g4.set_title('Time Components')

    threshold = 0.5
    prots_thresh = prots[((prots > threshold) | (prots < -threshold)).any(axis=1)]
    gene_thresh = geneMods[((geneMods > threshold) | (geneMods < -threshold)).any(axis=1)]

    Yp = sch.linkage(prots_thresh, method='centroid')
    Yg = sch.linkage(gene_thresh, method='centroid')
    Zp = sch.dendrogram(Yp, orientation='right', no_plot=True)
    Zg = sch.dendrogram(Yg, orientation='right', no_plot=True)
    index_p = Zp['leaves']
    index_g = Zg['leaves']
    prots_thresh = prots_thresh.iloc[index_p,:]
    gene_thresh = gene_thresh.iloc[index_g,:]

    g5 = sns.heatmap(prots_thresh, cmap="PiYG", center=0, yticklabels=True, vmin=-1, vmax=1, ax=ax[4])
    g5.set_yticklabels(g5.get_yticklabels(),rotation=0)
    g5.set_title(f'Proteins w/ component < or > than {threshold}')

    g6 = sns.heatmap(gene_thresh, cmap="PiYG", center=0, yticklabels=True, vmin=-1, vmax=1, ax=ax[5])
    g6.set_yticklabels(g6.get_yticklabels(),rotation=0)
    g6.set_title(f'Gene Modules w/ component < or > than {threshold}')

    return (g1, g2, g3, g4, g5, g6)

def generate_components_MEMA(tensor, comp, index0, index1, index2, ax, axnums):
    
    tfac = Decomposition(tensor, max_rr = comp + 1)
    tfac.perform_tfac()
    tfac = tfac.tfac[-1]
    
    facZero = pd.DataFrame(tfac.factors[0], columns=[f"{i}" for i in np.arange(1, tfac.rank + 1)], index=index0)
    facOne = pd.DataFrame(tfac.factors[1], columns=[f"{i}" for i in np.arange(1, tfac.rank + 1)], index=index1)
    facTwo = pd.DataFrame(tfac.factors[2], columns=[f"{i}" for i in np.arange(1, tfac.rank + 1)], index=index2)

    Yz = sch.linkage(facZero, method='centroid')
    Yo = sch.linkage(facOne, method='centroid')
    Yt = sch.linkage(facTwo, method='centroid')
    Zz = sch.dendrogram(Yz, orientation='right', no_plot=True)
    Zo = sch.dendrogram(Yo, orientation='right', no_plot=True)
    Zt = sch.dendrogram(Yt, orientation='right', no_plot=True)
    index_z = Zz['leaves']
    index_o = Zo['leaves']
    index_t = Zt['leaves']
    facZero = facZero.iloc[index_z,:]
    facOne = facOne.iloc[index_o,:]
    facTwo = facTwo.iloc[index_t,:]

    g0 = sns.heatmap(facZero, cmap="PiYG", center=0, yticklabels=True, vmin=-1, vmax=1, ax=ax[axnums[0]])
    g0.set_yticklabels(g0.get_yticklabels(),rotation=0)
    g0.set_title('Ligands')
    g1 = sns.heatmap(facOne, cmap="PiYG", center=0, yticklabels=True, vmin=-1, vmax=1, ax=ax[axnums[1]])
    g1.set_yticklabels(g1.get_yticklabels(),rotation=0)
    g1.set_title('ECM')
    g2 = sns.heatmap(facTwo, cmap="PiYG", center=0, yticklabels=True, vmin=-1, vmax=1, ax=ax[axnums[2]])
    g2.set_yticklabels(g2.get_yticklabels(),rotation=0)
    g2.set_title('Measurements')

    return g0, g1, g2
