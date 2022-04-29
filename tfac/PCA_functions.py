"""Contains function for performing PCA on OHSU data"""
from dataHelpers import Tensor_LINCS_MEMA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


def R2X_plot(datafile):
  R2X_list = []
  data = StandardScaler().Tensor_LINCS_MEMA(data)
  for x in range(2, 21):
      PCA_model = PCA(n_components=x)
      PCA_model.fit(data)
      R2X_list.append(np.sum(PCA_model.explained_variance_ratio_) * 100)
  plt.plot(range(2, 21), R2X_list)

def scores_plot(datafile):
    data = StandardScaler().Tensor_LINCS_MEMA(datafile)
    PCA_model = PCA(n_components=2)
    pca_scores = PCA_model.fit_transform(data)
    plt.scatter(pca_scores[:,0], pca_scores[:,1])

  