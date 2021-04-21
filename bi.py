import numpy as np

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml

def get_input_data_type(input_data):
    """入力するデータのtypeを確認

    Parameters
    ----------
    input_data : str or list
        可視化するデータ

    Returns
    -------
    str
        入力データの形に対応するモード
    """
    if type(input_data) is str:
        mode = 'csv'
    elif type(input_data) is list:
        mode = 'list'

    return mode

def histogram_visualize(input_data,save_path,column=None,kde=False):
    """ヒストグラムを作成する関数　データの次元は１次元のみに対応

    Parameters
    ----------
    input_data : csv path or list
        可視化するデータ
    save_path : str
        ファイルの保存先
    column : str, optional
        csvファイルの場合、可視化する列のコラム, by default None
    kde : bool, optional
        ヒストグラムにkdeを追加するかどうか, by default kde
    """
    mode = get_input_data_type(input_data)
    fig = plt.figure()
    if mode == 'csv' and column is not None:
        df = pd.read_csv(input_data)
        sns.histplot(df[column], kde=kde)
    elif mode == 'list':
        sns.histplot(input_data, kde=kde)

    plt.savefig(save_path)
    plt.clf()
    plt.close('all')

def scatter_visualize(x_data,y_data,):
    pass

def tsne(multi_vecs, dir):
    """多次元データを次元圧縮して2次元表示させる関数

    Parameters
    ----------
    multi_vecs : list
        ベクトルデータ
    dir : str
        保存先のディレクトリ
    """
    datas = []
    color = []
    dim = 0
    colorlist = ["r", "g", "b", "c", "m", "y", "k", "w"]

    for i, vecs in enumerate(multi_vecs.values()):
        for vec in vecs:
            dim = np.array(vec).shape[-1]
            datas.append(vec)
            color.append(colorlist[i])
    datas = np.array(datas).reshape((-1, dim))

    result = TSNE(n_components=2).fit_transform(datas)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for key, c in zip(multi_vecs.keys(), set(color)):
        same_c_datas = np.array(result)[np.array(color)==c]
        ax.scatter(same_c_datas[:,0], same_c_datas[:,1], c=c, label=key)
    ax.legend(loc='upper right')
    ax.set_xlabel('dim1')
    ax.set_ylabel('dim2')
    plt.savefig(dir)

if __name__=='__main__':
    test = np.zeros(2)
    print(test)