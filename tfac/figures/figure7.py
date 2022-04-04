""" Imputation error with varying missingness for each dataset
    Each plot is different dataset
    X axis, comp, Y axis, imputation error, each line is various missingness """

import pandas as pd
import numpy as np
from tensorpack import Decomposition
from ..dataHelpers import import_LINCS_MEMA
from .common import getSetup

def makeFigure():

    ax, f = getSetup((30, 30), (3, 3))
    MCF10A = import_LINCS_MEMA("mcf10a_ssc_Level4.tsv.xz")
    HMEC240 = import_LINCS_MEMA("hmec240l_ssc_Level4.tsv.xz")
    HMEC122 = import_LINCS_MEMA("hmec122l_ssc_Level4.tsv.xz")

    datasets = [MCF10A, HMEC240, HMEC122]
    c = 0
    for m in np.arange(0,3):
        for dataset in datasets:
            tensor = dataset.to_numpy()
            Decomp = Decomposition(tensor, max_rr=10)
            comps = np.arange(1,11)
            for drop_ in np.arange(1,4):
                Decomp.Q2X_chord(drop=drop_, repeat=10, mode=m)
                Q2X = Decomp.chordQ2X
                Q2Xmean = np.array(pd.DataFrame(Q2X).mean())
                Q2Xstd = np.array(pd.DataFrame(Q2X).std())
                ax[c].errorbar(x = comps, y = Q2Xmean, yerr = Q2Xstd, label=f'Dropped chords: {drop_}', capsize=5)
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
