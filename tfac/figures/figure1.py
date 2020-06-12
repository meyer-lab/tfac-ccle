"""
This creates Figure 1 - Variance vs AUC
"""
import numpy as np
import pandas as pd
import seaborn as sns
from tensorly.decomposition import parafac2
import tensorly as tl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from .figureCommon import subplotLabel, getSetup
from sklearn.metrics import roc_curve, roc_auc_score
from ..MRSA_dataHelpers import form_MRSA_tensor, get_patient_info

cohortID, outcomeID = get_patient_info()
outcome_bools = []

for outcome in outcomeID:
    if outcome == 'APMB':
        outcome_bools.append(0)
    else:
        outcome_bools.append(1)

outcomes = np.asarray(outcome_bools)
true_y = outcomes

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


    kf = KFold(n_splits=61)
    decisions = []
    for train, test in kf.split(patient_matrix):
        clf = LogisticRegression(random_state=1, max_iter=10000).fit(patient_matrix[train], outcomes[train])
        decisions.append(clf.decision_function(patient_matrix[test]))
    score_y = decisions
    auc = roc_auc_score(true_y, score_y)
    values_vars.append([variance, auc])
df_var = pd.DataFrame(values_vars)
df_var.columns = ['Variance', 'AUC']

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((10, 6), (1, 1))

    b = sns.pointplot(data=df_var, x='Variance', y='AUC', join=False, ax=ax[0])
    b.set_xlabel("Variance",fontsize=20)
    b.set_ylabel("AUC",fontsize=20)
    b.tick_params(labelsize=15)
    b.set_xticklabels(variance_list)
    ax[0].set_ylim(0, 1)

    
    # Add subplot labels
    subplotLabel(ax)

    return f
