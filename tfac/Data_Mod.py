import pandas as pd
import numpy as np
from dataHelpers import importLINCSprotein

def data_mod(x):
    df = importLINCSprotein()
    spec_df = ds.loc[(ds['Treatment'] == 'Control') | (ds['Treatment'] == x)]
    spec_df = spec_df.drop(columns = ['Sample description', 'Treatment', 'File'])
    y = spec_df.to_numpy()
    return y