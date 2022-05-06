from tensorly.metrics.similarity import correlation_index
from matplotlib.pyplot import savefig
import pandas as pd
import numpy as np
from tensorpack import Decomposition, impute
from ..dataHelpers import Tensor_LINCS_MEMA, reorder_table
from .common import getSetup
import seaborn as sns

def makeFigure():

    ax, f = getSetup((30, 15), (3, 1))
    MCF10A = Tensor_LINCS_MEMA("mcf10a_ssc_Level4.tsv.xz")
    HMEC240 = Tensor_LINCS_MEMA("hmec240l_ssc_Level4.tsv.xz")
    HMEC122 = Tensor_LINCS_MEMA("hmec122l_ssc_Level4.tsv.xz")

    repeat = 10
    dataset = MCF10A

    tensor = dataset.to_numpy()
    Decomp = Decomposition(tensor, max_rr=5)
    Decomp.perform_tfac()

    for count in range(0,repeat):

        tensor_c = np.moveaxis(tensor,m,0)
        impute.chord_drop(tensor_c, drop=100)
        tensor_c = np.moveaxis(tensor_c,0,m)
        Decomp_miss = Decomposition(tensor_c, max_rr=5)
        Decomp_miss.perform_tfac()

        factors = Decomp.tfac.factors
        factors_miss = Decomp_miss.tfac.factors

