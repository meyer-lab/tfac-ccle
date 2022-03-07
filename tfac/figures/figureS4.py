"""
Figure to show each components largest weights for MEMA Data
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup
from tensorpack import perform_CP
from tensorpack.cmtf import cp_normalize
from ..dataHelpers import import_LINCS_MEMA, proteinNames, reorder_table
from .figureS3 import plot_components

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((25, 10), (3, 6))

    HMEC240, ligand, ecm, meas = import_LINCS_MEMA("hmec240l_ssc_Level4.tsv.xz")

    plot_components(HMEC240, ligand, ecm, meas, ax)

    return f 
