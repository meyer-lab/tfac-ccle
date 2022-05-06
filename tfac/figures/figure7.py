""" Imputation error with varying missingness for each dataset
    Each plot is different dataset
    X axis, comp, Y axis, imputation error, each line is various missingness """

import pandas as pd
import numpy as np
from tensorpack import Decomposition
from ..dataHelpers import Tensor_LINCS_MEMA
from .common import getSetup

def makeFigure():

    ax, f = getSetup((15, 15), (3, 3))
    MCF10A = Tensor_LINCS_MEMA("mcf10a_ssc_Level4.tsv.xz")
    HMEC240 = Tensor_LINCS_MEMA("hmec240l_ssc_Level4.tsv.xz")
    HMEC122 = Tensor_LINCS_MEMA("hmec122l_ssc_Level4.tsv.xz")

    datasets = [MCF10A, HMEC240, HMEC122]
    c = 0
    for m in np.arange(0,3):
        for dataset in datasets:
            tensor = dataset.to_numpy()
            Decomp = Decomposition(tensor, max_rr=10)
            comps = np.arange(1,11)
            drop_ = 100
            Decomp.Q2X_chord(drop=drop_, repeat=20, mode=m)

            Q2X = Decomp.chordQ2X
            Q2Xmed = np.array(pd.DataFrame(Q2X).median())
            Q2Xiqr = np.array(pd.DataFrame(Q2X).apply(find_iqr))

            ax[c].errorbar(x = comps, y = Q2Xmed, yerr = Q2Xiqr, label=f'Dropped chords: {drop_}', capsize=5)
            ax[c].set_ylim(0,1)
            ax[c].set_xticks([x for x in comps])
            ax[c].set_xlabel("Number of components")
            ax[c].set_ylabel("Imputation accuracy")
            ax[c].legend()
            c += 1 

        ax[c-3].set_title(f"Imputation error for MCF10A along mode {m}")
        ax[c-2].set_title(f"Imputation error for HMEC240 along mode {m}")
        ax[c-1].set_title(f"Imputation error for HMEC112 along mode {m}")

    return f 

def find_iqr(x):
    return np.subtract(*np.percentile(x, [75, 25]))
