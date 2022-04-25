"""Contains function for importing and handling OHSU data"""
from os.path import join, dirname
import numpy as np
from sklearn.preprocessing import scale
import pandas as pd
import scipy.cluster.hierarchy as sch
import xarray as xa

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

def proteinNames():
    """Return protein names (data columns)"""
    data = importLINCSprotein()
    data = data.drop(columns=["Treatment", "Sample description", "File", "Time"], axis=1)
    return data.columns.values.tolist()

def process_proteins():
    """ import, group, and scale proteins into a dataframe, ready to be merge with RNAseq into a tensor. """
    df = importLINCSprotein()
    df.drop(columns=["Sample description", "File"], inplace=True)
    times = pd.unique(df["Time"])

    # Group replicates and average
    df = df.groupby(["Treatment", "Time"]).mean()

    for treatment in df.index.unique(level=0):
        df.loc[(treatment, 0), :] = df.loc[('Control', 0)].values

    df.drop('Control', inplace=True, level=0)
    df = df.sort_index()

    return df.apply(scale, axis=1, raw=True)

def importLINCSRNAseq(export=False):
    """ Import RNAseq data of MCF10A cells for growthfactor treatments from OHSU LINCS data. This function is only used in gene module analysis. """

    RNAseq = pd.read_csv(join(path_here, "tfac/data/ohsu/MDD_RNAseq_Level4.csv"), delimiter=",", index_col=0)

    row_avg = RNAseq.mean(axis=1)
    for indx in RNAseq.index:
        if row_avg[indx] <= 0.07:
            RNAseq.drop(indx, inplace=True)

    # column names
    cols = RNAseq.columns
    if export:
        RNAseq = RNAseq.apply(scale, axis=1, result_type='expand')
        RNAseq.columns = cols
        RNAseq.to_csv(join(path_here, "tfac/data/ohsu/RNAseq.txt"), sep='\t')

    return RNAseq

def process_RNAseq(df):
    """ Import gene modules, group, and scale into a dataframe, ready to be merged with the proteins data.
    Takes the dataframe of proteins as input for treatment labels. """

    # import the RNAseq data and pre-process
    RNAseq = pd.read_csv(join(path_here, "tfac/data/ohsu/module_expression.csv"), sep=',')
    RNAseq.rename(columns={"Unnamed: 0": "gene_modules"}, inplace=True)

    # Copy over control
    for treatment in df.index.unique(level=0):
        RNAseq[treatment + "_0"] = RNAseq["ctrl_0"]

    RNAseq = RNAseq.set_index("gene_modules").T
    RNAseq.index = RNAseq.index.str.split('_', expand=True)
    RNAseq.index = RNAseq.index.set_levels(RNAseq.index.levels[1].astype(int), level=1)

    RNAseq.drop('ctrl', inplace=True, level=0)
    RNAseq = RNAseq.reindex(index=df.index)

    return RNAseq.apply(scale, axis=1, raw=True)

def Tensor_LINCS_CCLE():
    """ Creates tensor in numpy array form and returns tensor, treatments, and time.
    Returns both the protein and RNAseq tensors in aligned format. """

    # import the proteins
    df = process_proteins()

    # import the RNAseq data
    RNAseq = process_RNAseq(df)

    # concatenate proteins and RNAseq data
    fullDF = pd.concat([df, RNAseq], axis=1)

    xdf = fullDF.to_xarray().transpose()

    return xdf.to_array().transpose()

def Tensor_LINCS_MEMA(datafile):
    """ Ligand, ECM, and phenotypic measurements of cells from LINCS MEMA dataset. """
    data = pd.read_csv(join(path_here, "tfac/data/ohsu/", datafile), index_col=["Ligand", "ECMp"], delimiter="\t", low_memory=False)

    missingCols = data.columns[data.isna().any()]
    assert len(missingCols) < 15
    data = data.dropna(axis=1)  # remove columns with no measurements
    data.drop(list(data.filter(regex='Conc')), axis=1, inplace=True)
    data.drop(list(data.filter(regex='Feret')), axis=1, inplace=True)
    data.drop(list(data.filter(regex='Orientation')), axis=1, inplace=True)
    data.drop(list(data.filter(regex='_SE')), axis=1, inplace=True)
    data.drop(list(data.filter(regex='LoessSCC')), axis=1, inplace=True)
    data = data.loc[:, data.dtypes == float]
    data.iloc[:, :] = scale(data)

    xdf = data.to_xarray().to_array(dim="Measurement")
    return xdf

def Tensor_LINCS_CycIF():
    """ Imports the cyclic immunofluorescence data from LINCS. """
    data = pd.read_csv(join(path_here, "tfac/data/CycIF/MDD_cycIF_Level4.csv"), delimiter=",", index_col=0) # data size: 660 x 36

    ctrl = data['ctrl_0']
    data.drop(columns='ctrl_0', inplace=True)
    data = data.T
    datacopy = data.copy()

    # to add the control at the start of each treatment
    datacopy.index = datacopy.index.str.split('_', expand=True) # split the treatment name and the time index
    for tr in list(datacopy.index.levels[0]):
        data = pd.concat([pd.DataFrame({tr+'_0': ctrl}).T, data])

    # split the indexes to the treatment and time indexes
    data.index = data.index.str.split('_', expand=True)
    data = data.sort_index(level=1) # sort with respect to the treatments
    data = data.loc[:, data.dtypes == float]
    data.iloc[:, :] = scale(data)

    data.index = pd.MultiIndex.from_tuples(data.index, names=["treatment", "time"]) # 42 x 660

    # a function to only split a string at a specific underscore; in this case we split at the third underscore
    def split_at(string, n):
        words = string.split('_')
        return '_'.join(words[:n]), '_'.join(words[n:])

    # split the data by staining and measurements

    new_cols = []
    for col in data.columns:
        new_cols.append(split_at(col, 3))
    data.columns = pd.MultiIndex.from_tuples(new_cols, names=["stains", "measures"])

    data = data.T.unstack(level=0)
    xdf = data.T.to_xarray().to_array('measures')

    return xdf

def reorder_table(df):
    """ Reorder a table's rows using heirarchical clustering. """
    # Reorder measurements based on similarity
    Y = sch.linkage(df.to_numpy(), method='centroid')
    index = sch.dendrogram(Y, orientation='right')['leaves']
    return df.iloc[index, :]

donorDict = {"1869": "RA",
            "1931": "RA",
            "2159": "RA",
            "2586": "N", 
            "2645": "N", 
            "2708": "RA", 
            "2759": "N"}

def process_RA_Tensor():
    """Structures all Rheumatoid Arthritis Synovial Fibroblast data into a usable tensor"""
    RA_df = pd.DataFrame()
    donor_list = ["1869", "1931", "2159", "2586", "2645", "2708", "2759"]
    rep_list = [1, 2]
    stimulants = ["IL1-a", "TNF-a", "PolyIC"]

    for donor in donor_list:
        for rep in rep_list:
            file_name = "tfac/data/RA_SF/SF_Donor_" + donor + "_" + donorDict[donor] + "_Rep" + str(rep) + ".csv"
            raw_data = pd.read_csv(join(path_here, file_name))
            avg_data = raw_data.groupby(['Sample Name']).mean().reset_index()
            
            for index, row in avg_data.iterrows():
                expString = row["Sample Name"].split(", ")
                if len(expString) == 2:
                    avg_data.loc[index, "Stimulant"] = expString[0]
                    avg_data.loc[index, "Inhibitor"] = expString[1]
                elif expString[0] == "bufferonly":
                    avg_data.loc[index, "Stimulant"] = "Buffer"
                    avg_data.loc[index, "Inhibitor"] = "Buffer"
                elif len(row["Sample Name"].split("_")) == 2:
                    avg_data.loc[index, "Stimulant"] = row["Sample Name"].split("_")[0]
                    avg_data.loc[index, "Inhibitor"] = "Spike"
            avg_data = avg_data.drop(['Sample Name', 'ExpPlate', 'LMXassayPlate'], axis=1)
            cytokines = avg_data.iloc[:, 0: -2].columns

            # Background (Spike and Otherwise) Subtraction
            for stimulant in stimulants:
                if stimulant in avg_data.columns:
                    avg_data.loc[(avg_data["Stimulant"] == stimulant)][stimulant] == np.nan
                spike_row = avg_data.loc[(avg_data["Stimulant"] == stimulant) & (avg_data["Inhibitor"] == "Spike")].reset_index().drop("index", axis=1).iloc[0, 0:-2]
                basal_row = avg_data.loc[(avg_data["Stimulant"] == "nostim") & (avg_data["Inhibitor"] == "noinh")].reset_index().drop("index", axis=1).iloc[0, 0:-2]
                for inh in avg_data.loc[(avg_data["Stimulant"] == stimulant)].Inhibitor.unique():
                    stim_inh_row = avg_data.loc[(avg_data["Stimulant"] == stimulant) & (avg_data["Inhibitor"] == inh)].iloc[0, 0:-2]
                    basal_spike_df = pd.concat([stim_inh_row - spike_row, basal_row], axis=1).transpose()
                    avg_data.loc[(avg_data["Stimulant"] == stimulant) & (avg_data["Inhibitor"] == inh), cytokines] = basal_spike_df.max().to_frame().transpose().values
            avg_data[cytokines] = np.log(avg_data[cytokines].values)
            avg_data[cytokines] -= avg_data.loc[avg_data.Stimulant == "Buffer", cytokines].values
            avg_data[cytokines] = avg_data[cytokines].clip(lower=0)
            
            avg_data["Donor"] = donor
            avg_data["Replicate"] = rep
            avg_data["Status"] = donorDict[donor]
            avg_data = pd.melt(avg_data, id_vars=["Stimulant", "Inhibitor", "Donor", "Status", "Replicate"], value_vars=cytokines).rename({"variable": "Cytokine", "value": "Log MFI"}, axis=1)
            avg_data = avg_data.loc[(avg_data.Stimulant != "Buffer") & (avg_data.Inhibitor != "Spike")]
            RA_df = pd.concat([RA_df, avg_data], axis=0)

    # Average Over Replicates
    RA_df = RA_df.groupby(["Stimulant", "Inhibitor", "Donor", "Status", "Cytokine"])["Log MFI"].mean().reset_index()
    RA_df.to_csv("RA_DataFrame.csv")
    return RA_df


def make_RA_Tensor():
    """Processes RA DataFrame into Xarray Tensor"""
    RA_df = pd.read_csv("RA_DataFrame.csv")
    stimulants = RA_df.Stimulant.unique()
    inhibitors = RA_df.Inhibitor.unique()
    cytokines = RA_df.Cytokine.unique()
    donors = RA_df.Donor.unique()

    tensor = np.empty((len(stimulants), len(inhibitors), len(cytokines), len(donors)))
    tensor[:] = np.nan
    for i, stim in enumerate(stimulants):
        for j, inh in enumerate(inhibitors):
            for k, cyto in enumerate(cytokines):
                for ii, donor in enumerate(donors):
                    if stim != inh:
                        entry = RA_df.loc[(RA_df.Stimulant == stim) & (RA_df.Inhibitor == inh) & (RA_df.Cytokine == cyto) & (RA_df.Donor == donor)]["Log MFI"].values
                        tensor[i, j, k, ii] = np.mean(entry)
    # Normalize
    for i, _ in enumerate(cytokines):
        tensor[:, :, i, :][~np.isnan(tensor[:, :, i, :])] /= np.nanmax(tensor[:, :, i, :])

    RA_xarray = xa.DataArray(tensor, dims=("Stimulant", "Inhibitor", "Cytokine", "Donor"), coords={"Stimulant": stimulants, "Inhibitor": inhibitors, "Cytokine": cytokines, "Donor": donors})
    RA_xarray.to_netcdf("RA Tensor DataSet.nc")
    return tensor
