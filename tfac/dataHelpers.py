'''Contains function for importing data from and sending data to synapse'''
import os
import numpy as np
import pandas as pd
import tqdm
import h5py
from synapseclient import Synapse, File
from .dataProcess import normalize

path = os.path.dirname(os.path.abspath(__file__))


def importData(username, password, dataType=None):
    '''Data Import from synapse
    ----------------------------------------------
    Parameters:
        username: string
            Synapse username
        password: string
            Synapse password
        data: string
            'Copy Number', 'Methylation', or 'Gene Expression'

    Returns:
        df: DataFrame
            Data from the CCLE in data frame format
    '''

    # Input Checking
    if dataType is None:
        print('Invalid Data Set')
        print('Enter:', 'Copy Number All,', 'Methylation All,', 'or Gene Expression All')
        return None
    syn = Synapse()
    try:
        syn.login(username, password)
    except BaseException:
        print('Bad Username or Password')
        return None

    # Find Data
    if dataType == 'Copy Number All':
        data = syn.get('syn21089502')
    elif dataType == 'Methylation All':
        data = syn.get('syn21089540')
    elif dataType == 'Gene Expression All':
        data = syn.get('syn21089539')
    elif dataType == 'Copy Number':
        data = syn.get('syn21303730')
    elif dataType == 'Methylation':
        data = syn.get('syn21303732')
    elif dataType == 'Gene Expression':
        data = syn.get('syn21303731')

    df = pd.read_csv(data.path, index_col=0, header=0)
    syn.logout()
    return df


def exportData(username, password, data, nm):
    '''Pandas Data Frame to upload back to synapse as an excel file
    ---------------------------------------------------------------
    Parameters:
        username: String
            Your synapse username
        password: String
            You synapse password
        data: Data Frame
            Pandas object containing the data to upload to synapse as an excel file
        name: String
            A name for the file in synapse
    '''
    syn = Synapse()
    syn.login(username, password)
    proj = syn.get('syn21032722')
    data.to_csv('data/file.csv')
    syn.store(File(path='data/file.csv', name=nm, parent=proj))
    syn.logout()


def makeTensor(username, password, impute=False):
    '''Generate correctly aligned tensor for factorization'''
    syn = Synapse()
    syn.login(username, password)

    # Setup Data Carriers
    copy_number = pd.DataFrame()
    methylation = pd.DataFrame()
    gene_expression = pd.DataFrame()

    # Get Data
    for chunk1 in tqdm.tqdm(pd.read_csv(syn.get('syn21303730').path, chunksize=150), ncols=100, total=87):
        copy_number = pd.concat((copy_number, chunk1))
    for chunk2 in tqdm.tqdm(pd.read_csv(syn.get('syn21303732').path, chunksize=150), ncols=100, total=87):
        methylation = pd.concat((methylation, chunk2))
    for chunk3 in tqdm.tqdm(pd.read_csv(syn.get('syn21303731').path, chunksize=150), ncols=100, total=87):
        gene_expression = pd.concat((gene_expression, chunk3))

    arr = normalize(np.stack((gene_expression.values[:, 1:], copy_number.values[:, 1:], methylation.values[:, 1:])))
    if impute:
        arr = np.nan_to_num(arr)

    # Create final tensor
    syn.logout()
    return arr


def getCellLineComps(imputed=False, rank=100):
    '''Import cell line components'''
    if imputed:
        if rank == 100:
            filename = os.path.join(path, './data/Imputed_Components_100.hdf5')
        else:
            filename = os.path.join(path, './data/Imputed_Components_50.hdf5')
        with h5py.File(filename, 'r') as f:
            data = f["Cell_Line_Comps"][:]
            f.close()
        return data
    else:
        filename = os.path.join(path, './data/HDF5/cell_comps_25.hdf5')
        with h5py.File(filename, 'r') as f:
            data = f["comps"][:]
            f.close()
        return data.T


def getGeneComps(imputed=False):
    '''Import gene components --- rank 25 cp'''
    if imputed:
        if rank == 100
            filename = os.path.join(path, './data/Imputed_Components_100.hdf5')
        else:
            filename = os.path.join(path, './data/Imputed_Components_50.hdf5')
        with h5py.File(filename, 'r') as f:
            data = f["Gene_Comps"][:]
            f.close()
        return data
    else:
        filename = os.path.join(path, './data/HDF5/gene_comps_25.hdf5')
        with h5py.File(filename, 'r') as f:
            data = f["comps"][:]
            f.close()
        return data.T


def getCharacteristicComps():
    '''Import characteristic components --- rank 25 cp'''
    filename = os.path.join(path, './data/HDF5/measurement_comps_25.hdf5')
    with h5py.File(filename, 'r') as f:
        data = f["comps"][:]
        f.close()
    return data.T


def cellLineNames():
    """Get a Full List of Cell Lines for a plot legend
    ------------------------------------------------------------
    ***Calling np.unique(ls) yields the 23 different cancer types***
    """
    filename = os.path.join(path, "./data/cellLines(aligned,precut).csv")
    df = pd.read_csv(filename)
    names = np.insert(df.values, 0, "22RV1_PROSTATE")
    ls = [x.split('_', maxsplit=1)[1] for x in names]
    return ls
