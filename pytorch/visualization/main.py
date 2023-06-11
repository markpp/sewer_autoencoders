import numpy as np
import sys
import os
import cv2

import matplotlib.pyplot as plt
import seaborn as sns
import json
import umap
from sklearn.decomposition import PCA
import pandas as pd


def pca_visualization(z, y):
    n_components = 8
    pca = PCA(n_components=n_components)
    transform = pca.fit(z)
    print("explained_variance: {}".format(pca.explained_variance_))
    print("explained_variance_ratio: {}".format(pca.explained_variance_ratio_))
    #print("mean: {}".format(pca.mean_))
    print("noise_variance: {}".format(pca.noise_variance_))
    transformed = transform.transform(z)

    df = pd.DataFrame()
    df['1st dim.'] = transformed[:,0]
    df['2nd dim.'] = transformed[:,1]

    #sns.violinplot(data=df, x=y, y=transformed[:,1])

    sns.scatterplot(data=df, x="1st dim.", y="2nd dim.", hue=y)
    plt.show()

def umap_visualization(z, y):

    fit = umap.UMAP(n_neighbors=8,
                    min_dist=0.8,
                    n_components=2,
                    metric='euclidean')
    u = fit.fit_transform(z)

    df = pd.DataFrame()
    df['1st dim.'] = u[:,0]
    df['2nd dim.'] = u[:,1]
    sns.scatterplot(data=df, x="1st dim.", y="2nd dim.", hue=y)
    plt.show()

if __name__ == '__main__':
    """
    Test script

    Command:
        python test.py
    """

    selection = np.random.randint(512, size=100)

    y = np.load("y.npy")[selection]
    z = np.load("z.npy")[selection]

    pca_visualization(z, y)
    #umap_visualization(z, y)
