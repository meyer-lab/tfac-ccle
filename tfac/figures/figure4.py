"""
This creates Figure 4.
"""
from .figureCommon import subplotLabel, getSetup
from .figure1 import proteinScatterPlot
from ..Data_Mod import form_tensor, z_score_tensor_bymeasure, z_score_tensor_byprotein
from ..tensor import cp_decomp, reorient_factors

tensor, treatments, times = form_tensor()
tensor_z = z_score_tensor_byprotein(tensor)
results = cp_decomp(tensor_z, 8)
comps = reorient_factors(results[1])


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((15, 20), (7, 4))

    counter = 0
    for i in range(8):
        for j in range(i + 1, 8):
            proteinScatterPlot(ax[counter], comps[2], i+1, j+1)
            counter += 1
    
    # Add subplot labels
    subplotLabel(ax)

    return f
