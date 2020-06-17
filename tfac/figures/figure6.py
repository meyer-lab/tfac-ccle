"""
This creates Figure 6 - ROC Curve.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from tensorly.decomposition import parafac2
from sklearn.metrics import roc_curve, roc_auc_score
from .figureCommon import subplotLabel, getSetup
from ..MRSA_dataHelpers import form_MRSA_tensor, get_patient_info, produce_outcome_bools, find_CV_decisions

_, outcomeID = get_patient_info()

true_y = produce_outcome_bools(outcomeID)

tensor_slices, cytokines, geneIDs = form_MRSA_tensor(1)
components = 38
parafac2tensor = None
best_error = np.inf
for run in range(1):
    decomposition, errors = parafac2(tensor_slices, components, return_errors=True, tol=1e-7, n_iter_max=1000)
    if best_error > errors[-1]:
        best_error = errors[-1]
        parafac2tensor = decomposition

patient_matrix = parafac2tensor[1][2]

score_y = find_CV_decisions(patient_matrix, true_y)

fpr, tpr, thresholds = roc_curve(true_y, score_y)
auc = roc_auc_score(true_y, score_y)
print(auc)

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((8, 8), (1, 1))

    df = pd.DataFrame()
    df['FPR'] = fpr
    df['TPR'] = tpr
    sns.lineplot(data=df, x='FPR', y='TPR', estimator=None, ax=ax[0])
    df = pd.DataFrame()
    df['FPR'] = [0, 1]
    df['TPR'] = [0, 1]
    sns.lineplot(data=df, x='FPR', y='TPR', estimator=None, ax=ax[0])

    # Add subplot labels
    subplotLabel(ax)

    return f
