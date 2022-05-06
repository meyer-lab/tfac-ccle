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

    m = 0
    repeat = 10
    dataset = HMEC240

    tensor = dataset.to_numpy()
    Decomp = Decomposition(tensor, max_rr=5)
    Decomp.perform_tfac()

    for count in range(0,repeat):

        tensor_c = np.moveaxis(tensor,m,0)
        impute.chord_drop(tensor_c, drop=100)
        tensor_c = np.moveaxis(tensor_c,0,m)
        Decomp_miss = Decomposition(tensor_c, max_rr=5)
        Decomp_miss.perform_tfac()

        factors = Decomp.tfac[-1].factors
        factors_miss = Decomp_miss.tfac[-1].factors

        fZero = np.zeros((dataset.shape[0], 5, repeat))
        fOne = np.zeros((dataset.shape[1], 5, repeat))
        fTwo = np.zeros((dataset.shape[2], 5, repeat))

        fZero[:,:,count] = factors[0] - factors_miss[0]
        fOne[:,:,count] = factors[1] - factors_miss[1]
        fTwo[:,:,count] = factors[2] - factors_miss[2]
        
    fZero_avg = np.mean(fZero, axis=2)
    fOne_avg = np.mean(fOne, axis=2)
    fTwo_avg = np.mean(fTwo, axis=2)

    
    labels = [f"Cmp. {i}" for i in np.arange(1, fZero_avg.shape[1] + 1)]
    fac = pd.DataFrame(fZero_avg, columns=labels, index=dataset.coords[dataset.dims[0]])
    fac = reorder_table(fac)
    g1 = sns.heatmap(fac.T, ax=ax[0], cmap="PRGn", center=0)
    g1.set_title(dataset.dims[0])

    labels = [f"Cmp. {i}" for i in np.arange(1, fOne_avg.shape[1] + 1)]
    fac = pd.DataFrame(fOne_avg, columns=labels, index=dataset.coords[dataset.dims[1]])
    fac = reorder_table(fac)
    g2 = sns.heatmap(fac.T, ax=ax[1], cmap="PRGn", center=0)
    g2.set_title(dataset.dims[1])

    labels = [f"Cmp. {i}" for i in np.arange(1, fTwo_avg.shape[1] + 1)]
    fac = pd.DataFrame(fTwo_avg, columns=labels, index=dataset.coords[dataset.dims[2]])
    fac = reorder_table(fac)
    g1 = sns.heatmap(fac.T, ax=ax[2], cmap="PRGn", center=0)
    g1.set_title(dataset.dims[2])

    return f
