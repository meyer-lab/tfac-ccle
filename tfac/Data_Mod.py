import pandas as pd
import numpy as np
from .dataHelpers import importLINCSprotein

def data_mod(x, df = None):
    '''Creates a slice of the data tensor corresponding to the inputted treatment'''
    if type(df) != pd.core.frame.DataFrame:
        df = importLINCSprotein()
    spec_df = df.loc[(df['Treatment'] == 'Control') | (df['Treatment'] == x)]
    times = spec_df['Time'].to_numpy().tolist()
    spec_df = spec_df.drop(columns = ['Sample description', 'Treatment', 'File', 'Time'])
    y = spec_df.to_numpy()
    return y, spec_df, times

def form_tensor():
    '''Creates tensor in numpy array form and returns tensor, treatments, and time'''
    df = importLINCSprotein()

    unique_treatments = np.unique(df['Treatment'].values).tolist()
    unique_treatments.remove('Control')

    slices = []
    df_slices = []
    for treatment in unique_treatments:
        array, df_array, times = data_mod(treatment, df)
        slices.append(array)
    tensor = np.vstack(slices)
    return tensor, unique_treatments, times