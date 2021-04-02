"""
Functions for reverse projection.
"""
import numpy as np
import pandas as pd


def get_Flattened_Matrices(result, data, treatment_list, times):
    # create a 5x42 DataFrame of decompsed component values
    flattened = [pd.DataFrame(dd).T for dd in result[0]]

    df = pd.concat(flattened, axis=1)

    # component values for control at T=0
    ctrl = df.iloc[:, 0]

    # Remove all times that aren't 24 or 48
    df.drop([0, 1, 2, 3], inplace=True, axis=1)

    # add control back in
    ctrl = pd.DataFrame(ctrl)
    df = pd.concat([ctrl, df], axis=1)

    # rename columns
    cols = [str(x) + "_" + str(y) for x in treatment_list for y in times[-2:]]
    cols = ["ctrl_0"] + cols
    df.columns = cols

    # sort by column name
    df = df.reindex(sorted(df.columns), axis=1)
    data = data.reindex(sorted(data.columns), axis=1)

    # make dataframes into numpy arrays
    df = df.to_numpy()
    data = data.to_numpy()
    return df, data


def get_reconstruct(P, X):
    Ppinv = np.linalg.pinv(P.T)
    return Ppinv, np.matmul(Ppinv, X.T)


def find_factors(result, data, treatment_list, times):
    P, X = get_Flattened_Matrices(result, data, treatment_list, times)
    Ppinv, W = get_reconstruct(P, X)
    return P, X, Ppinv, W

