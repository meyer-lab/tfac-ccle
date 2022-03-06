""" import drug response data. """
from os.path import join, dirname
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import h5py
from tensorpack.figureCommon import getSetup
from tensorpack.decomposition import Decomposition
from tensorpack.cmtf import perform_CP, reorient_factors
from tensorpack.plot import tfacr2x, reduction

path_here = dirname(dirname(__file__))
drugs = ["BEZ235", "Trametinib", "5FU", "AZD5438", "Panobinostat", "MG132", "Everolimus", "JQ1", "Bortezomib", "MK1775", "Cabozantinib"]
conc = ["control", "conc 1", "conc 2", "conc 3", "conc 4", "conc 5", "conc 6","conc 7"]
time = np.linspace(0.0, 94.0, 189)
g1g2 = ["G1", "S/G2"]

def import_data():
    """
    Shape = (11, 8, 189, 2)
    11: number of drugs
    8: conditions / concentrations
    189: timepoints
    2: G1/G2
    """
    f = h5py.File(join(path_here, "data/drug_response/data.jld"), "r")
    d = np.array(f["data"][:])

    return d - np.mean(d)

def makeFigure():
    ax, f = getSetup((9, 6), (2, 3))
    data = import_data()
    alldat = Decomposition(data)
    alldat.perform_tfac()
    alldat.perform_PCA()
    tfacr2x(ax[0], alldat)
    reduction(ax[1], alldat)

    tFac = perform_CP(data, r=3)
    tFac = reorient_factors(tFac)
    drug = pd.DataFrame(tFac.factors[0], columns=[f"Cmp. {i}" for i in np.arange(1, tFac.rank + 1)], index=drugs)
    concentrations = pd.DataFrame(tFac.factors[1], columns=[f"Cmp. {i}" for i in np.arange(1, tFac.rank + 1)], index=conc)
    times = pd.DataFrame(tFac.factors[2], columns=[f"Cmp. {i}" for i in np.arange(1, tFac.rank + 1)], index=time)
    G1SG2 = pd.DataFrame(tFac.factors[3], columns=[f"Cmp. {i}" for i in np.arange(1, tFac.rank + 1)], index=g1g2)

    g1 = sns.heatmap(drug, cmap="PRGn", center=0, xticklabels=True, yticklabels=True, cbar=True, vmin=-1.0, vmax=1.0, ax=ax[2])
    g1.set_yticklabels(g1.get_yticklabels(), rotation = 0)
    g2 = sns.heatmap(concentrations, cmap="PRGn", center=0, yticklabels=True, cbar=True, vmin=-1.0, vmax=1.0, ax=ax[3])
    g2.set_yticklabels(g2.get_yticklabels(), rotation = 0)
    ax[4].plot(time, times)
    ax[4].set_xlabel("Time")
    ax[4].set_ylabel("Component Weight")
    ax[4].legend(["Cmp. 1", "Cmp. 2", "Cmp. 3", "Cmp. 4", "Cmp. 5"])
    g4 = sns.heatmap(G1SG2, cmap="PRGn", center=0, yticklabels=True, cbar=True, vmin=-1.0, vmax=1.0, ax=ax[5])
    g4.set_yticklabels(g4.get_yticklabels(), rotation = 0)
    return f