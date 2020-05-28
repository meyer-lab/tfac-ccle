'''Contains function for importing data from and sending data to synapse'''
from os.path import join, dirname
import pandas as pd
from tensorly.metrics.regression import variance as tl_var

path_here = dirname(dirname(__file__))


def importLINCSprotein():
    """ Import protein characterization from LINCS. """
    dataA = pd.read_csv(join(path_here, "tfac/data/01_Laura_Heiser__Sean_Gross_A.csv"))
    dataB = pd.read_csv(join(path_here, "tfac/data/01_Laura_Heiser__Sean_Gross_B.csv"))
    dataC = pd.read_csv(join(path_here, "tfac/data/01_Laura_Heiser__Sean_Gross_C.csv"))

    dataA["File"] = "A"
    dataB["File"] = "B"
    dataC["File"] = "C"

    return pd.concat([dataA, dataB, dataC])


def compProteins(comps):
    """Returns the top three weighted proteins for each component in input protein component matrix"""
    i = np.shape(comps)  # input tensor decomp output
    proteins = proteinNames()
    proteinNum, compNum = np.shape(comps[i[0] - 1])
    compName = []
    topProtein = []

    for x in range(0, compNum):
        compName.append('Col' + str(x + 1))

    dfComps = pd.DataFrame(data=comps[i[0] - 1], index=proteins, columns=compName)
    for y in range(0, compNum):
        topProtein.append(compName[y])
        rearranged = dfComps.sort_values(by=compName[y], ascending=False)
        rearrangedNames = list(rearranged.index.values)
        for z in range(0, 3):
            topProtein.append(rearrangedNames[z])

    return topProtein


def proteinNames():
    data = importLINCSprotein()
    data = data.drop(columns=['Treatment', 'Sample description', 'File', 'Time'], axis=1)
    proteinN = data.columns.values.tolist()
    return proteinN

def form_MRSA_tensor():
    dfClin, dfCoh = importClinicalMRSA()
    dfCyto = clinicalCyto(dfClin, dfCoh)
    dfCyto = dfCyto.sort_values(by='sid')
    dfCyto = dfCyto.set_index('sid')
    cytokines = dfCyto.columns
    
    dfExp = importExpressionData()
    dfExp = dfExp.T
    geneIDs = dfExp.iloc[0, 0:].to_list()
    dfExp.columns = geneIDs
    dfExp = dfExp.drop('Geneid')
    
    cytoNumpy = dfCyto.to_numpy().T
    expNumpy = dfExp.to_numpy().T
    
    expNumpy = expNumpy.astype(float)
    var = (tl_var(expNumpy)/tl_var(cytoNumpy))
    cytoNumpy = cytoNumpy * 29
    
    tensor_slices = [cytoNumpy, expNumpy]

    return tensor_slices, cytokines, geneIDs

def importClinicalMRSA():
    """import clincal MRSA data"""
    dataClin = pd.read_csv(join(path_here, "tfac/data/mrsa/mrsa_s1s2_clin+cyto_073018.csv"))
    dataCohort = pd.read_csv(join(path_here, "tfac/data/mrsa/clinical_metadata_cohort1.txt"), delimiter='\t')
    return dataClin, dataCohort

def clinicalCyto(dataClinical, dataCohort):
    """isolate cytokine data from clinical"""
    rowSize, colSize = dataClinical.shape
    patientID = list(dataClinical["sid"])
    
    dataClinical = dataClinical.drop(dataClinical.iloc[:, 0:3], axis=1)
    dataClinical = dataClinical.drop(dataClinical.iloc[:, 1:207], axis=1)
    
    """isolate patient IDs from cohort 1"""
    dataCohort = dataCohort.drop(columns=['age','gender','race','sampletype','pair','outcome_txt'], axis=1)
    cohortID = list(dataCohort["sample"])
    IDSize, column = dataCohort.shape
   
    cytokineData = pd.DataFrame()
    
    for y in range(0, rowSize):
        for z in range(0, IDSize):
            if (cohortID[z]).find(str(patientID[y])) != -1:
                temp = dataClinical.loc[dataClinical['sid']==patientID[y]]
                cytokineData = pd.concat([temp, cytokineData])
    cytokineData.sort_values(by=['sid'])
    return cytokineData

def importExpressionData():
    """import expression data"""
    df = pd.read_table(join(path_here, "tfac/data/mrsa/expression_counts_cohort1.txt"))
    df.drop(["Chr", "Start", "End", "Strand", "Length"], inplace=True, axis=1)
    return df


###################################################### CCLE DATA FUNCTIONS ##########################################################
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
        print('For Raw Data Enter:', '"Copy Number All",', '"Methylation All",', 'or "Gene Expression All"')
        print('For Processed Data Enter:', '"Copy Number"', '"Methylation"', 'or "Gene Expression"')
        return None
    syn = Synapse()
    try:
        syn.login(username, password)
    except BaseException:
        print('Bad Username or Password')
        return None

    # Find Data -- TODO: FIGURE OUT WHAT THESE ALL SPECIFICALLY REPRESENT
    if dataType == 'Copy Number All':
        df = pd.read_excel(syn.get('syn21033823').path)
    elif dataType == 'Methylation All':
        df = pd.read_excel(syn.get('syn21033929').path)
    elif dataType == 'Gene Expression All':
        df = pd.read_excel(syn.get('syn21033805').path)
    elif dataType == 'Copy Number':
        df = pd.read_csv(syn.get('syn21303730').path, index_col=0, header=0)
    elif dataType == 'Methylation':
        df = pd.read_csv(syn.get('syn21303732').path, index_col=0, header=0)
    elif dataType == 'Gene Expression':
        df = pd.read_csv(syn.get('syn21303731').path, index_col=0, header=0)

    syn.logout()
    return df


def makeTensor(username, password):
    '''Generate correctly aligned tensor for factorization'''
    syn = Synapse()
    syn.login(username, password)

    # Get Data
    copy_number = importData(username, password, 'Copy Number')
    methylation = importData(username, password, 'Methylation')
    gene_expression = importData(username, password, 'Gene Expression')

    # Create final tensor
    arr = normalize(np.stack((gene_expression.values, copy_number.values, methylation.values)))

    syn.logout()
    return arr


def cellLineNames():
    """Get a Full List of Cell Lines for a plot legend
    ------------------------------------------------------------
    ***Calling np.unique(ls) yields the 23 different cancer types***
    """
    filename = join(path_here, "./data/cellLines(aligned,precut).csv")
    df = pd.read_csv(filename)
    names = np.insert(df.values, 0, "22RV1_PROSTATE")
    ls = [x.split('_', maxsplit=1)[1] for x in names]
    return ls


def geneNames():
    '''Get a full list of the ordered gene names in the tensor (names are EGID's)'''
    genes = importData("robertt", "LukeKuechly59!", "Gene Expression")
    return np.array(genes.index)
