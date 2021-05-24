import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

def main():
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

if __name__=='__main__':
    main()