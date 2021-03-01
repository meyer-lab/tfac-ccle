"""
This creates Figure 2. This figure includes the Partial Tucker Treatment/Component Heat Map.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from ..Data_Mod import form_tensor
from ..dataHelpers import importLINCSprotein, get_flipped_tucker


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    
    tensor, treatment_list, times = form_tensor()

    result = get_flipped_tucker(tensor, 5)

    compList = ['1', '2', '3', '4', '5']
    trmtTimeList = []
    for i, name in enumerate(treatment_list):
        for t in range(len(times)):
            trmtTimeList.append(name + '_' + str(times[t]))
    trmtTime = np.reshape(trmtTimeList, (7, 6))
    trmtMap = pd.DataFrame()
    for y in range(len(trmtTime)):
        temp = pd.DataFrame(data=result[0][y], index=trmtTime[y], columns=compList)
        trmtMap = trmtMap.append(temp)
        palette = sns.cubehelix_palette(len(trmtTime), light=.8, dark=.2, reverse=True, start=1, rot=-2)
    dict = {}
    time_tick = []
    for x in range(7):
        for y in range(6):
            dict[str(trmtTime[x, y])] = palette[x]
            time_tick.append(times[y])
    colors = pd.Series(trmtTimeList, index=trmtMap.index).map(dict)
    f = sns.clustermap(trmtMap, row_cluster=False, col_cluster=False, row_colors=colors, cmap='PiYG', center=0, yticklabels=time_tick)
    for val, label in enumerate(trmtTime[:, 0]):
        f.ax_col_dendrogram.bar(0, 0, color=dict[label], label=treatment_list[val], linewidth=0)
    f.ax_col_dendrogram.legend(loc="center left", ncol=5)
    f.fig.suptitle('Treatment-Time vs. Component')
    return f
