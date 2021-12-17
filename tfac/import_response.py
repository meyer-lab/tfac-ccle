""" import drug response data. """
from os.path import join, dirname
import numpy as np
import h5py
from tensorpack.figureCommon import getSetup
from tensorpack.decomposition import Decomposition
from tensorpack.plot import tfacr2x, reduction

path_here = dirname(dirname(__file__))

def import_data():
    """
    Shape = (11, 8, 189, 2)
    11: number of drugs
    8: conditions / concentrations
    189: timepoints
    2: G1/G2
    """
    f = h5py.File(join(path_here, "tfac/data/drug_response/data.jld"), "r")
    d = np.array(f["data"][:])
    return d - np.mean(d)


def figure():
    ax, f = getSetup((5, 8), (3, 2))
    alldat = Decomposition(import_data())
    alldat.perform_tfac()
    alldat.perform_PCA()
    tfacr2x(ax[0], alldat)
    reduction(ax[1], alldat)
    g1 = Decomposition(import_data()[:, :, :, 0])
    g2 = Decomposition(import_data()[:, :, :, 1])
    g1.perform_tfac()
    g1.perform_PCA()
    g2.perform_tfac()
    g2.perform_PCA()
    tfacr2x(ax[2], g1)
    reduction(ax[3], g1)
    tfacr2x(ax[4], g2)
    reduction(ax[5], g2)
    return f