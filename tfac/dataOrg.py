import numpy as np
import pandas as pd

def extractData(filename, columns = None, row = 0, col = None):
    return pd.read_excel(filename, header = row, index_col = col, usecols = columns)