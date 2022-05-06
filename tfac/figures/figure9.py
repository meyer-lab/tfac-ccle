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

    