""" Imputation error """

import pandas as pd
import numpy as np
from tensorpack import Decomposition
from ..dataHelpers import Tensor_LINCS_MEMA
from .common import getSetup

def makeFigure():

    ax, f = getSetup((15, 5), (1, 3))
    MCF10A = Tensor_LINCS_MEMA("mcf10a_ssc_Level4.tsv.xz")
    HMEC240 = Tensor_LINCS_MEMA("hmec240l_ssc_Level4.tsv.xz")
    HMEC122 = Tensor_LINCS_MEMA("hmec122l_ssc_Level4.tsv.xz")

    imputation(MCF10A, ax=ax[0])

    return f 

def find_iqr(x):
    return np.subtract(*np.percentile(x, [75, 25]))

def imputation(dataset, ax):
    for m in np.arange(0,3):
        tensor = dataset.to_numpy()
        Decomp = Decomposition(tensor, max_rr=10)
        comps = np.arange(1,11)
        drop_ = 100
        Decomp.Q2X_chord(drop=drop_, repeat=10, mode=m)

        Q2X = Decomp.chordQ2X
        Q2Xmed = np.array(pd.DataFrame(Q2X).median())
        Q2Xiqr = np.array(pd.DataFrame(Q2X).apply(find_iqr))

        ax.errorbar(x = comps, y = Q2Xmed, yerr = Q2Xiqr, label=f'Mode: {m}', capsize=5)
        ax.set_ylim(0,1)
        ax.set_xticks([x for x in comps])
        ax.set_xlabel("Number of components")
        ax.set_ylabel("Imputation accuracy")
        ax.set_title(f"{dataset}")
        ax.legend()