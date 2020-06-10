"""
This creates Figure 2
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
from ..dataHelpers import form_MRSA_tensor, get_patient_info

cohortID, outcomeID = get_patient_info()


outcome_bools = []

for outcome in outcomeID:
    if outcome == 'APMB':
        outcome_bools.append(0)
    else:
        outcome_bools.append(1)

outcomes = np.asarray(outcome_bools)
true_y = outcomes

values_comps = []
for components in range(1, 39):
    tensor_slices, cytokines, geneIDs = form_MRSA_tensor(1)
    parafac2tensor = None
    best_error = np.inf
    for run in range(4):
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
    values_comps.append([components, auc])
df_comp = pd.DataFrame(values_comps)
df_comp.columns = ['Components', 'AUC']

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((10, 5), (1, 1))
    b = sns.scatterplot(data=df_comp, x='Components', y='AUC', ax=ax[0])
    b.set_xlabel("Components",fontsize=20)
    b.set_ylabel("AUC",fontsize=20)
    b.tick_params(labelsize=14)
    ax[0].set_ylim(0, 1)
    
    # Add subplot labels
    subplotLabel(ax)

    return f
