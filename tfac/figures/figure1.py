"""
This creates Figure 1 - Variance vs AUC
"""
import numpy as np
import pandas as pd
import seaborn as sns
from tensorly.decomposition import parafac2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from .figureCommon import subplotLabel, getSetup
from ..MRSA_dataHelpers import form_MRSA_tensor, get_patient_info, produce_outcome_bools

_, outcomeID = get_patient_info()

true_y = produce_outcome_bools(outcomeID)

values_vars = []
variance_list = [.001, .1, 1, 2.25, 5, 29, 100, 1000, 100000]
components = 38
for variance in variance_list:
    tensor_slices, cytokines, geneIDs = form_MRSA_tensor(variance)
    parafac2tensor = None
    best_error = np.inf
    for run in range(1):
        decomposition, errors = parafac2(tensor_slices, components, return_errors=True)
        if best_error > errors[-1]:
            best_error = errors[-1]
            parafac2tensor = decomposition

    patient_matrix = parafac2tensor[1][2]


    auc = find_CV_AUC(patient_matrix, true_y)
    values_vars.append([variance, auc])
df_var = pd.DataFrame(values_vars)
df_var.columns = ['Variance', 'AUC']

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((10, 6), (1, 1))

    b = sns.pointplot(data=df_var, x='Variance', y='AUC', join=False, ax=ax[0])
    b.set_xlabel("Variance", fontsize=20)
    b.set_ylabel("AUC", fontsize=20)
    b.tick_params(labelsize=15)
    b.set_xticklabels(variance_list)
    ax[0].set_ylim(0, 1)

    # Add subplot labels
    subplotLabel(ax)

    return f
