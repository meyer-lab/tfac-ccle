import numpy as np
import pandas as pd

def extractData(filename, columns):
    return pd.read_excel(filename, header = 2, index_col = 0, usecols = columns)