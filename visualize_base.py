import sklearn.base
import bhtsne
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


class BHTSNE(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, dimensions=2, perplexity=30.0, theta=0.5, rand_seed=-1):
        self.dimensions = dimensions
        self.perplexity = perplexity
        self.theta = theta
        self.rand_seed = rand_seed

    def fit_transform(self, x):
        return bhtsne.tsne(
            x.astype(np.float64), dimensions=self.dimensions, perplexity=self.perplexity, theta=self.theta,
            rand_seed=self.rand_seed)

if __name__=='__main__':
    from sklearn.datasets import load_wine
    wine = load_wine()

    tsne = BHTSNE()
    tsne_res = tsne.fit_transform(wine.data)

    plt.scatter(tsne_res[:,0],tsne_res[:,1])
    plt.show()