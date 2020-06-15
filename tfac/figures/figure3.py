"""
This creates Figure 3. - Components vs AUC
"""
import numpy as np
import pandas as pd
import seaborn as sns
from tensorly.decomposition import parafac2
from sklearn.metrics import roc_auc_score
from .figureCommon import subplotLabel, getSetup
from ..MRSA_dataHelpers import form_MRSA_tensor, get_patient_info, find_CV_decisions, produce_outcome_bools

_, outcomeID = get_patient_info()
true_y = produce_outcome_bools(outcomeID)

values_comps = []
for components in range(1, 39):
    tensor_slices, cytokines, geneIDs = form_MRSA_tensor(1)
    parafac2tensor = None
    best_error = np.inf
    for run in range(1):
        decomposition, errors = parafac2(tensor_slices, components, return_errors=True, n_iter_max=1000)
        if best_error > errors[-1]:
            best_error = errors[-1]
            parafac2tensor = decomposition

    patient_matrix = parafac2tensor[1][2]
    
    score_y = find_CV_decisions(patient_matrix)
    auc = roc_auc_score(true_y, score_y)
    values_comps.append([components, auc])
df_comp = pd.DataFrame(values_comps)
df_comp.columns = ['Components', 'AUC']

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((10, 5), (1, 1))
    b = sns.scatterplot(data=df_comp, x='Components', y='AUC', ax=ax[0])
    b.set_xlabel("Components", fontsize=20)
    b.set_ylabel("AUC", fontsize=20)
    b.tick_params(labelsize=14)
    ax[0].set_ylim(0, 1)

    # Add subplot labels
    subplotLabel(ax)

    return f
