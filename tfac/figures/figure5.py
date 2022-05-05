""" factorize the HMEC122L ECM data. """

import xarray as xa
import tensorly as tl
import numpy as np
import seaborn as sns
import pandas as pd
from tensorly.decomposition import parafac
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from copy import copy
from ..dataHelpers import Tensor_LINCS_MEMA, process_RA_Tensor, make_RA_Tensor
from ..plotHelpers import plot_heatmaps
from .common import getSetup, subplotLabel


def makeFigure():
    """ make heatmaps of factors when decomposed individually. """

    ax, f = getSetup((12, 14), (4, 2), multz={6: 1})
    subplotLabel(ax)
    process_RA_Tensor()
    make_RA_Tensor()
    num_comps = 5

    RA_data = xa.open_dataarray("RA Tensor DataSet.nc")
    tFacAllM = factorTensor(RA_data.values, numComps=num_comps)
    tFacAllM.normalize()
    R2Xplot(ax[0], RA_data.values, compNum=15)
    plot_tFac_RA(ax[1], tFacAllM, RA_data, "Stimulant", numComps=num_comps)
    plot_tFac_RA(ax[2], tFacAllM, RA_data, "Inhibitor", numComps=num_comps)
    plot_tFac_RA(ax[3], tFacAllM, RA_data, "Donor", numComps=num_comps)
    plot_tFac_RA(ax[4], tFacAllM, RA_data, "Cytokine", numComps=num_comps)
    RA_LogReg_plot(ax[5], tFacAllM, RA_data, numComps=num_comps)
    acc_scan(ax[6], RA_data)
    #tensor = Tensor_LINCS_MEMA("hmec122l_ssc_Level4.tsv.xz")
    #plot_heatmaps(tensor, ax)

    return f


def factorTensor(tensor, numComps):
    """ Takes Tensor, and mask and returns tensor factorized form. """
    tfac = parafac(np.nan_to_num(tensor), rank=numComps, mask=np.isfinite(tensor), n_iter_max=5000, tol=1e-9)
    tensor = tensor.copy()
    tensor[np.isnan(tensor)] = tl.cp_to_tensor(tfac)[np.isnan(tensor)]
    return tfac


def R2Xplot(ax, tensor, compNum):
    """Creates R2X plot for non-neg CP tensor decomposition"""
    varHold = np.zeros(compNum)
    for i in range(1, compNum + 1):
        tFac = factorTensor(tensor, i)
        varHold[i - 1] = calcR2X(tensor, tFac)

    ax.scatter(np.arange(1, compNum + 1), varHold, c='k', s=20.)
    ax.set(title="R2X", ylabel="Variance Explained", xlabel="Number of Components", ylim=(0, 1), xlim=(0, compNum + 0.5), xticks=np.arange(0, compNum + 1))


def calcR2X(tensorIn, tensorFac):
    """ Calculate R2X. """
    tErr = np.nanvar(tl.cp_to_tensor(tensorFac) - tensorIn)
    return 1.0 - tErr / np.nanvar(tensorIn)


def plot_tFac_RA(ax, tFac, RA_Array, mode, numComps=3):
    """Plots tensor factorization of cells"""
    mode_labels = RA_Array[mode]
    coord = RA_Array.dims.index(mode)
    mode_facs = tFac[1][coord]
    tFacDF = pd.DataFrame()
    for i in range(0, numComps):
        tFacDF = pd.concat([tFacDF, pd.DataFrame({"Component_Val": mode_facs[:, i], "Component": (i + 1), mode: mode_labels})])

    tFacDF = pd.pivot(tFacDF, index="Component", columns=mode, values="Component_Val")
    if mode == "Donor":
        tFacDF = tFacDF[[1869, 1931, 2159, 2708, 2586, 2645, 2759]]
    sns.heatmap(data=tFacDF, ax=ax)
 

def RA_LogReg_plot(ax, tFac, RA_Array, numComps):
    """Plot factor weights for donor RA prediction"""
    coord = RA_Array.dims.index("Donor")
    mode_facs = tFac[1][coord]
    Donor_RA_y = [1, 1, 1, 0, 0, 1, 0]

    LR_RA = LogisticRegression(random_state=0).fit(mode_facs, Donor_RA_y)
    RA_comp_weights = pd.DataFrame({"Component": np.arange(1, numComps + 1), "Coefficient": LR_RA.coef_[0]})
    print(LR_RA.predict(mode_facs))
    print(LR_RA.score(mode_facs, Donor_RA_y))
    sns.barplot(data=RA_comp_weights, x="Component", y="Coefficient", color="k", ax=ax)
    SVC_RC = SVC().fit(mode_facs, Donor_RA_y)
    print(SVC_RC.score(mode_facs, Donor_RA_y))


def acc_scan(ax, RA_data):
    """Scans to see how many components are required for accurate RA identification"""
    num_comps = np.arange(10, 21)
    accDF = pd.DataFrame()
    Donor_RA_y = [1, 1, 1, 0, 0, 1, 0]
    for num_comp in num_comps:
        tFacAllM = factorTensor(RA_data.values, numComps=num_comp)
        tFacAllM.normalize()
        coord = RA_data.dims.index("Donor")
        mode_facs = tFacAllM[1][coord]
        LR_RA = LogisticRegression(random_state=0).fit(mode_facs, Donor_RA_y)
        SVC_RA = SVC().fit(mode_facs, Donor_RA_y)
        accDF = pd.concat([accDF, pd.DataFrame({"Num Comps": [num_comp], "Accuracy": SVC_RA.score(mode_facs, Donor_RA_y), "Model": "SVC"})])
        accDF = pd.concat([accDF, pd.DataFrame({"Num Comps": [num_comp], "Accuracy": LR_RA.score(mode_facs, Donor_RA_y), "Model": "LR"})])

    sns.barplot(data=accDF, x="Num Comps", y="Accuracy", hue="Model", ax=ax)
