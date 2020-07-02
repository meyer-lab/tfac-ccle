"""
This creates Figure 3. - Components vs AUC
"""
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score
from .figureCommon import subplotLabel, getSetup
from ..MRSA_dataHelpers import get_patient_info, find_CV_decisions, produce_outcome_bools
from ..tensor import MRSA_decomposition

_, outcomeID = get_patient_info()
true_y = produce_outcome_bools(outcomeID)
variance = .007
reps = []
values_comps = []
for _ in range(3):
    for components in range(1, 39):
        tensor_slices, parafac2tensor = MRSA_decomposition(variance, components)
        patient_matrix = parafac2tensor[1][2]

        score_y = find_CV_decisions(patient_matrix, true_y, C=10)
        auc = roc_auc_score(true_y, score_y)
        values_comps.append([components, auc])
    df_comp = pd.DataFrame(values_comps)
    df_comp.columns = ['Components', 'AUC']
    reps.append(df_comp)
reps = pd.concat(reps)


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((15, 8), (1, 1))
    b = sns.pointplot(data=reps, x='Components', y='AUC', ax=ax[0], s=70, join=False) # blue
    b.set_xlabel("Components",fontsize=20)
    b.set_ylabel("AUC",fontsize=20)
    b.tick_params(labelsize=14)
    ax[0].set_ylim(0, 1)

    # Add subplot labels
    subplotLabel(ax)

    return f
