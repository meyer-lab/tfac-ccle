"""Data pre-processing and tensor formation"""
import pandas as pd
import numpy as np
from .dataHelpers import importLINCSprotein


def form_tensor():
    """Creates tensor in numpy array form and returns tensor, treatments, and time"""
    df = importLINCSprotein()
    df.drop(columns=["Sample description", "File"], inplace=True)
    times = pd.unique(df["Time"])

    # Group replicates and average
    df = df.groupby(["Treatment", "Time"]).mean()

    for treatment in df.index.unique(level=0):
        df.loc[(treatment, 0), :] = df.loc[('Control', 0)].values

    df.drop('Control', inplace=True)
    df = df.sort_index()
    
    dfArray = df.to_numpy()
    tensor = np.reshape(dfArray, (-1, len(times), dfArray.shape[1]))

    return tensor, df.index.unique(level=0), times
