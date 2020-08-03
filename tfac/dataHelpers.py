"""Contains function for importing and handling OHSU data"""
from os.path import join, dirname
import numpy as np
import pandas as pd

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


def ohsu_data():
    """ Import OHSU data for PARAFAC2"""
    atac = pd.read_csv(join(path_here, "tfac/data/ohsu/MDD_ATACseq_Level4.csv"))
    cycIF = pd.read_csv(join(path_here, "tfac/data/ohsu/MDD_cycIF_Level4.csv"))
    GCP = pd.read_csv(join(path_here, "tfac/data/ohsu/MDD_GCP_Level4.csv"))
    IF = pd.read_csv(join(path_here, "tfac/data/ohsu/MDD_IF_Level4.csv"))
    L1000 = pd.read_csv(join(path_here, "tfac/data/ohsu/MDD_L1000_Level4.csv"))
    RNAseq = pd.read_csv(join(path_here, "tfac/data/ohsu/MDD_RNAseq_Level4.csv"))
    RPPA = pd.read_csv(join(path_here, "tfac/data/ohsu/MDD_RPPA_Level4.csv"))
    return atac, cycIF, GCP, IF, L1000, RNAseq, RPPA


def compProteins(comps):
    """Returns the top three weighted proteins for each component in input protein component matrix"""
    i = np.shape(comps)  # input tensor decomp output
    proteins = proteinNames()
    _, compNum = np.shape(comps[i[0] - 1])
    compName = []
    topProtein = []

    for x in range(0, compNum):
        compName.append("Col" + str(x + 1))

    dfComps = pd.DataFrame(data=comps[i[0] - 1], index=proteins, columns=compName)
    for y in range(0, compNum):
        topProtein.append(compName[y])
        rearranged = dfComps.sort_values(by=compName[y], ascending=False)
        rearrangedNames = list(rearranged.index.values)
        for z in range(0, 3):
            topProtein.append(rearrangedNames[z])

    return topProtein


def proteinNames():
    """Return protein names (data columns)"""
    data = importLINCSprotein()
    data = data.drop(columns=["Treatment", "Sample description", "File", "Time"], axis=1)
    proteinN = data.columns.values.tolist()
    return proteinN


def printOutliers(results):
    """Prints most extremem protein outliers of partial tucker decomposition of OHSU data based on IQR"""
    df = pd.DataFrame(results[1][0])
    proteins = importLINCSprotein()
    columns = proteins.columns[3:298]
    df["Proteins"] = columns
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    prots = {}
    for i in range(df.columns.size - 1):
        print("Component", str(i + 1), "1.5*IQR:", np.round((Q1[i] - 1.5 * IQR[i]), 2), np.round((Q3[i] + 1.5 * IQR[i]), 2))
        positives = []
        negatives = []
        for _, col in df.iterrows():
            if col[i] < (Q1[i] - 1.5 * IQR[i]):
                negatives.append((col[i], col["Proteins"]))
                if col["Proteins"] not in prots:
                    prots[col["Proteins"]] = 1
                else:
                    prots[col["Proteins"]] += 1
            elif col[i] > (Q3[i] + 1.5 * IQR[i]):
                positives.append((col[i], col["Proteins"]))
                if col["Proteins"] not in prots:
                    prots[col["Proteins"]] = 1
                else:
                    prots[col["Proteins"]] += 1
        print()
        negatives = sorted(negatives)[:7]
        positives = sorted(positives)[-7:]
        for tup in positives:
            print(tup[1])
        for tup in positives:
            print(np.round(tup[0], 2))
        print()
        for tup in negatives:
            print(tup[1])
        for tup in negatives:
            print(np.round(tup[0], 2))
        print()
    print(prots)
def find_CV_decisions(patient_matrix, outcomes, n_splits=61, random_state=None):
    kf = KFold(n_splits=n_splits)
    decisions = []
    for train, test in kf.split(patient_matrix):
        clf = LogisticRegression(random_state=random_state, max_iter=10000).fit(patient_matrix[train], outcomes[train])
        decisions.append(clf.decision_function(patient_matrix[test]))
    score_y = decisions
    return score_y


def produce_outcome_bools(statusID):
    """Returns a list of booleans for progressor/resolver status ready to use for logistic regression"""
    outcome_bools = []
    for outcome in statusID:
        if outcome == 'APMB':
            outcome_bools.append(0)
        else:
            outcome_bools.append(1)

    return np.asarray(outcome_bools)


def get_patient_info():
    """Return specific patient ID information"""
    dataCohort = pd.read_csv(join(path_here, "tfac/data/mrsa/clinical_metadata_cohort1.txt"), delimiter="\t")
    cohortID = list(dataCohort["sample"])
    statusID = list(dataCohort["outcome_txt"])

    return cohortID, statusID


def form_MRSA_tensor(variance):
    """Create list of data matrices for parafac2"""
    dfClin, dfCoh = importClinicalMRSA()
    dfCyto = clinicalCyto(dfClin, dfCoh)
    dfCyto = dfCyto.sort_values(by="sid")
    dfCyto = dfCyto.set_index("sid")
    cytokines = dfCyto.columns

    dfExp = importExpressionData()
    dfExp = dfExp.T
    geneIDs = dfExp.iloc[0, 0:].to_list()
    dfExp.columns = geneIDs
    dfExp = dfExp.drop("Geneid")

    cytoNumpy = dfCyto.to_numpy().T
    expNumpy = dfExp.to_numpy().T

    expNumpy = expNumpy.astype(float)
    cytoNumpy = cytoNumpy * variance

    tensor_slices = [cytoNumpy, expNumpy]

    return tensor_slices, cytokines, geneIDs


def importClinicalMRSA():
    """import clincal MRSA data"""
    dataClin = pd.read_csv(join(path_here, "tfac/data/mrsa/mrsa_s1s2_clin+cyto_073018.csv"))
    dataCohort = pd.read_csv(join(path_here, "tfac/data/mrsa/clinical_metadata_cohort1.txt"), delimiter="\t")
    return dataClin, dataCohort


def clinicalCyto(dataClinical, dataCohort):
    """isolate cytokine data from clinical"""
    rowSize, _ = dataClinical.shape
    patientID = list(dataClinical["sid"])

    dataClinical = dataClinical.drop(dataClinical.iloc[:, 0:3], axis=1)
    dataClinical = dataClinical.drop(dataClinical.iloc[:, 1:206], axis=1)

    # isolate patient IDs from cohort 1
    dataCohort = dataCohort.drop(columns=["age", "gender", "race", "sampletype", "pair", "outcome_txt"], axis=1)
    cohortID = list(dataCohort["sample"])
    IDSize, _ = dataCohort.shape

    cytokineData = pd.DataFrame()

    for y in range(0, rowSize):
        for z in range(0, IDSize):
            if (cohortID[z]).find(str(patientID[y])) != -1:
                temp = dataClinical.loc[dataClinical["sid"] == patientID[y]]
                cytokineData = pd.concat([temp, cytokineData])
    cytokineData.sort_values(by=["sid"])
    return cytokineData


def importExpressionData():
    """import expression data"""
    df = pd.read_table(join(path_here, "tfac/data/mrsa/expression_counts_cohort1.txt"))
    df.drop(["Chr", "Start", "End", "Strand", "Length"], inplace=True, axis=1)
    return df