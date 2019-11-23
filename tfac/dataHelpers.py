'''Contains function for importing data from and sending data to synapse'''
import numpy as np
import pandas as pd
import tqdm
from synapseclient import Synapse, File


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

    ## Input Checking
    if dataType is None:
        print('Invalid Data Set')
        print('Enter:', 'Copy Number All,', 'Methylation All,', 'or Gene Expression All')
        return None
    syn = Synapse()
    try:
        syn.login(username, password)
    except:
        print('Bad Username or Password')
        return None

    ## Find Data
    if data == 'Copy Number All':
        data = syn.get('syn21089502')
    elif data == 'Methylation All':
        data = syn.get('syn21089540')
    elif data == 'Gene Expression All':
        data = syn.get('syn21089539')

    df = pd.read_excel(data.path, index_col=0)
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
    syn.store(File(path='file.csv', name=nm, parent=proj))
    syn.logout()

def makeTensor(username, password):
    '''Generate correctly aligned tensor for factorization'''
    syn = Synapse()
    syn.login(username, password)
    
    ## Setup Data Carriers
    copy_number = pd.DataFrame()
    methylation = pd.DataFrame()
    gene_expression = pd.DataFrame()

    ## Get Data
    for chunk in tqdm.tqdm(pd.read_csv(syn.get('syn21303730').path, chunksize=150), ncols=100, total=87):
        copy_number = pd.concat((copy_number, chunk))
    for chunk in tqdm.tqdm(pd.read_csv(syn.get('syn21303732').path, chunksize=150), ncols=100, total=87):
        methylation = pd.concat((copy_number, chunk))
    for chunk in tqdm.tqdm(pd.read_csv(syn.get('syn21303731').path, chunksize=150), ncols=100, total=87):
        gene_expression = pd.concat((copy_number, chunk))

    ## Create final tensor
    syn.logout()
    return gene_expression, copy_number, methylation
