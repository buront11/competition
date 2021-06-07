import pandas as pd
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils import data

def category_visualize():
    df = pd.read_csv('data/advitisement/train.tsv', delimiter='\t')
    df = df.dropna(how='any')
    column = list(range(5,1559))
    df_category = df.iloc[:,column]

    catecory_info = []

    for index, row in df_category.iterrows():
        value = sum(row)
        if value:
            catecory_info.append(1)
        else:
            catecory_info.append(0)

    df['category'] = catecory_info

    df_label = df['Y'].values.tolist()

    class_list = []
    for cate, label in zip(catecory_info, df_label):
        if cate == 1 and label == 0:
            class_list.append('cate no ad')
        elif cate == 1 and label == 1:
            class_list.append('cate  ad')
        elif cate == 0 and label == 0:
            class_list.append('no cate no ad')
        else:
            class_list.append('no cate  ad')

    df['class'] = class_list

    sns.countplot(x='class', data=df)
    plt.show()

    print(df_category)

def area_visualize():
    df = pd.read_csv('data/advitisement/train.tsv', delimiter='\t')
    df = df.dropna(how='any')

    df['area'] = df['width'] * df['height']
    print(df)

    sns.scatterplot('area', 'Y', data=df)
    plt.show()

def heatmap():
    df = pd.read_csv('data/advitisement/train.tsv', delimiter='\t')
    df = df.dropna(how='any')
    df['area'] = df['width'] * df['height']
    column = list(range(5,1559))
    df_category = df.iloc[:,column]

    catecory_info = []

    for index, row in df_category.iterrows():
        value = sum(row)
        if value:
            catecory_info.append(value)
        else:
            catecory_info.append(0)

    df['category'] = catecory_info

    analysis_columns = ['width', 'height', 'area', 'aratio', 'local', 'category', 'Y']
    df = df.loc[:,analysis_columns]
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='summer')
    plt.show()

if __name__=='__main__':
    heatmap()