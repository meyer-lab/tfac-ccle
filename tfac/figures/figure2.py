"""
This creates Figure 2. This figure includes the Partial Tucker Treatment/Component Heat Map.
"""
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from ..dataHelpers import form_tensor
from tensorpac.ccle import perform_CMTF


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    ax, f = getSetup((8, 4), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    tensor, rTensor, _, times = form_tensor()
    result = perform_CMTF(tensor, rTensor)

    sns.heatmap(result.factors[0], cmap="PiYG", ax=ax[0])
    ax[1].plot(times, result.factors[1])

    return f
