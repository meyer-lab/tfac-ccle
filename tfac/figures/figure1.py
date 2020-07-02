"""
This creates Figure 1 - Variance vs AUC
"""
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score
from .figureCommon import subplotLabel, getSetup
from ..MRSA_dataHelpers import get_patient_info, produce_outcome_bools, find_CV_decisions
from ..tensor import MRSA_decomposition

_, outcomeID = get_patient_info()

true_y = produce_outcome_bools(outcomeID)

values_vars = []
variance_list = [.0001, .001, .005, .007, .01, .05, .5]
components = [4, 9, 14, 19, 24, 29, 34, 37]
df_vars = []
for component in components:
    for variance in variance_list:
        tensor_slices, parafac2tensor = MRSA_decomposition(variance, component)
        patient_matrix = parafac2tensor[1][2]

        score_y = find_CV_decisions(patient_matrix, true_y, C=10)
        auc = roc_auc_score(true_y, score_y)
        values_vars.append([variance, auc])
    df_var = pd.DataFrame(values_vars)
    df_var.columns = ['Variance', 'AUC']
    df_vars.append(df_var)


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((18, 20), (4, 2))
    for idx, comp in enumerate(components):
        b = sns.pointplot(data=df_vars[comp], x='Variance', y='AUC', join=False, ax=ax[idx])
        b.set_title(str(components[comp]) + " Components", fontsize=25)
        b.set_xlabel("Cytokine Weighting", fontsize=20)
        b.set_ylabel("AUC", fontsize=20)
        b.tick_params(labelsize=15)
        b.set_xticklabels(variance_list)
        ax[idx].set_ylim(0, 1)

    # Add subplot labels
    subplotLabel(ax)

    return f
