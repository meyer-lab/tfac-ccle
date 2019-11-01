'''Contains function for importing data from and sending data to synapse'''

import pandas as pd
from synapseclient import Synapse


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

    df = pd.read_excel(data.path, index_col = 0)
    syn.logout()
    return df
