'''
Contains functions having to do wth tensor decomposition
'''

import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.decomposition import parafac, tucker

def perform_parafac(tens, rank):
    '''Run Canonical Polyadic Decomposition on a tensor
    ---------------------------------------------------------
    Parameters:
        tens: numpy tensor
            Data tensor with which to perform factorization
        rank: int
            Number of component vectors desired along each axis during factorization
    
    Returns:
        factors: list of numpy arrays (length: dimensionality of the tensor)
            List of arrays containing the components for each axis (i.e. Factors[0] is the array containing the components for the first axis)
            
    '''
    output = parafac(tens, rank)
    return output.factors
