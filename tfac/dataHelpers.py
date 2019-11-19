'''Contains function for importing data from and sending data to synapse'''

import pandas as pd
from synapseclient import Synapse, File
from pybiomart.dataset import Dataset

dataset = Dataset(name='hsapiens_gene_ensembl', host='http://www.ensembl.org')

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
    
def convertID():
    '''converts all ids to EGID and returns an excel file'''
    data = extractData('data/geneIdentifiers.xlsx', 'A:D')
    data = data.to_numpy()
    ranges = np.array([21338, 56202, 23316])

    methylation = np.append(data[:12158,0:2],data[12159:21338,0:2], axis = 0).astype(str)
    geneExp = data[:,2].astype(str)
    copyNum = data[:23316,3].astype(str)
    
    geneID = [methylation, geneExp, copyNum]
    
    