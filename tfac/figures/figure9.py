from tensorly.metrics.similarity import correlation_index
import pandas as pd
import numpy as np
from tensorpack import Decomposition, impute
from ..dataHelpers import Tensor_LINCS_MEMA, reorder_table
from .common import getSetup
import seaborn as sns

def makeFigure():

    ax, f = getSetup((15, 5), (1, 3))
    MCF10A = Tensor_LINCS_MEMA("mcf10a_ssc_Level4.tsv.xz")
    HMEC240 = Tensor_LINCS_MEMA("hmec240l_ssc_Level4.tsv.xz")
    HMEC122 = Tensor_LINCS_MEMA("hmec122l_ssc_Level4.tsv.xz")

    compare_corr(MCF10A, ax[0])
    ax[0].set_title("MCF10A")
    compare_corr(HMEC240, ax[1])
    ax[1].set_title("HMEC240")
    compare_corr(HMEC122, ax[2])
    ax[2].set_title("HMEC122")

    return f
    
def compare_corr(dataset, ax):

    repeat = 20
    tensor = dataset.to_numpy()
    Decomp = Decomposition(tensor, max_rr=8)
    comps = np.arange(1,9)
    Decomp.perform_tfac()
    corr_index = np.zeros((3,repeat,8))

    for m in range(0,3):
        for count in range(0,repeat):

            tensor_c = np.moveaxis(tensor,m,0)
            impute.chord_drop(tensor_c, drop=100)
            tensor_c = np.moveaxis(tensor_c,0,m)
            Decomp_miss = Decomposition(tensor_c, max_rr=8)
            Decomp_miss.perform_tfac()
            factors = Decomp.tfac
            factors_miss = Decomp_miss.tfac

            for c,(f,f_m) in enumerate(zip(factors, factors_miss)):
                corr = correlation_index(f.factors, f_m.factors)
                corr_index[m,count,c] = corr

    corr_avg = np.median(corr_index, axis=1)
    for m in range(0,3):
        ax.plot(comps, corr_avg[m,:], label=f'Mode: {m}')
        ax.set_ylim(0,1)
        ax.set_xticks([x for x in comps])
        ax.set_xlabel("Number of components")
        ax.set_ylabel("Correlation Index")
        ax.legend()
