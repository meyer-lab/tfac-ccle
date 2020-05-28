"""
This creates Figure 5.
"""
import numpy as np
from .figureCommon import subplotLabel, getSetup
from ..dataHelpers import form_MRSA_tensor
from tensorly.decomposition import parafac2
import tensorly as tl
from tensorly.parafac2_tensor import parafac2_to_slice
from tensorly.metrics.regression import variance as tl_var


tl.set_backend("numpy")


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    
    tensor_slices, cytokines, geneIDs = form_MRSA_tensor()
    
    print(tl_var(tensor_slices[1])/tl_var(tensor_slices[0]))
    
    AllR2X = []
    for i in range(1, 11):
        parafac2tensor = parafac2(tensor_slices, i, random_state=1)
        AllR2X.append(R2Xparafac2(tensor_slices, parafac2tensor))
    
    print(AllR2X)

    return tensor_slices

def R2Xparafac2(tensor_slices, decomposition):
    R2X = [0, 0]
    for idx, tensor_slice in enumerate(tensor_slices):
        reconstruction = parafac2_to_slice(decomposition, idx, validate=False)
        R2X[idx] = 1.0 - tl_var(reconstruction - tensor_slice) / tl_var(tensor_slice)
    return R2X