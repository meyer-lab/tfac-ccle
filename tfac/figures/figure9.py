""" Imputation error """

import pandas as pd
import numpy as np
from tensorpack import Decomposition
from ..dataHelpers import Tensor_LINCS_MEMA
from .common import getSetup

def makeFigure():

    ax, f = getSetup((5, 5), (1, 1))
    MCF10A = Tensor_LINCS_MEMA("mcf10a_ssc_Level4.tsv.xz")
    HMEC240 = Tensor_LINCS_MEMA("hmec240l_ssc_Level4.tsv.xz")
    HMEC122 = Tensor_LINCS_MEMA("hmec122l_ssc_Level4.tsv.xz")

    imputation(MCF10A, ax=ax[0])

    return f 

def find_iqr(x):
    return np.subtract(*np.percentile(x, [75, 25]))

def imputation(dataset, ax):
    tensor = dataset.to_numpy()
    Decomp = Decomposition(tensor, max_rr=6)
    comps = np.arange(1,7)

    Decomp.Q2X_chord(drop=100, repeat=5, mode=0)
    Q2X = Decomp.chordQ2X
    Q2Xmed = np.array(pd.DataFrame(Q2X).median())
    Q2Xiqr = np.array(pd.DataFrame(Q2X).apply(find_iqr))

    ax.errorbar(x = comps, y = Q2Xmed, yerr = Q2Xiqr, label=f'Mode: {0}', capsize=5)
    ax.set_ylim(0,1)
    ax.set_xticks([x for x in comps])
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Imputation accuracy")
    ax.legend()

    return None