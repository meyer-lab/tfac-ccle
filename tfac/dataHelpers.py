"""Contains function for importing and handling OHSU data"""
from os.path import join, dirname
import numpy as np
import pandas as pd

path_here = dirname(dirname(__file__))


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
   
