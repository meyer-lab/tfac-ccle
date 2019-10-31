'''Contains function for importing data from and sending data to synapse'''

import pandas as pd
from synapseclient import Synapse


def importData(username, password, data=None):
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
    if dataType == 'Copy Number All':
        data = syn.get('syn21080087')
    elif dataType == 'Methylation All':
        data = syn.get('syn21080130')
    elif dataType == 'Gene Expression All':
        data = syn.get('syn21080124')

    df = pd.read_excel(data.path)
    syn.logout()
    return df
