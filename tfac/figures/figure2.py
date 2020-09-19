"""
This creates Figure 2. This figure includes heat maps for Partial Tucker Protein/Component and Treatment/Component.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from ..tensor import partial_tucker_decomp, flip_factors
from ..Data_Mod import form_tensor
from ..dataHelpers import importLINCSprotein

component = 5
tensor, treatment_list, times = form_tensor()
pre_flip_result = partial_tucker_decomp(tensor, [2], component)

result = flip_factors(pre_flip_result)

compList = ['1', '2', '3', '4', '5']
trmtTimeList = []
for i, name in enumerate(treatment_list):
    for t in range(len(times)):
        trmtTimeList.append(name + '_' + str(times[t]))
trmtTime = np.reshape(trmtTimeList, (7, 6))
trmtMap = pd.DataFrame()
for y in range(len(trmtTime)):
    temp = pd.DataFrame(data = result[0][y], index = trmtTime[y], columns = compList)
    trmtMap = trmtMap.append(temp)

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    row = 1
    col = 2
    ax, f = getSetup((12, 7), (row, col))
    heatMap(trmtMap, "Treatment by Component", ax[0])
    subplotLabel(ax)
    return f


def heatMap(df, map_title, ax):
    ax.title(map_title, fontsize=12)
    sns.heatmap(df, cmap = 'RdYlGn', linewidths=0.10, xticklabels = df.columns, yticklabels = df.index, ax=ax)
    