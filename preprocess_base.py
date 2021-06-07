import joblib
from matplotlib.pyplot import sca
import pandas as pd
from pandas.core.tools.datetimes import Scalar
from pandas.io.parsers import ParserBase

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset

class PreprocessBase():
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def standard_scaler(self, columns=None):
        scaler = StandardScaler()
        if columns:
            scaler.fit(self.train_data.loc[:,columns])
            self.train_data.loc[:,columns] = scaler.transform(self.train_data.loc[:,columns])
            self.test_data.loc[:,columns] = scaler.transform(self.test_data.loc[:,columns])
        else:
            scaler.fit(self.train_data)
            self.train_data = pd.DataFrame(scaler.transform(self.train_data), columns=self.train_data.columns)
            self.test_data = pd.DataFrame(scaler.transform(self.test_data), columns=self.test_data.columns)


def load_tsv(tsv_path):
    df = pd.read_csv(tsv_path, delimiter='\t')
    return df

def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df

def drop_nan(df):
    df = df.dropna(how='any')
    return df

def tensor_dataset(data, label):
    return TensorDataset(data, label)

if __name__=='__main__':
    from sklearn.datasets import load_wine
    wine = load_wine()

    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    train, test = train_test_split(df, train_size=0.8)

    print(train)
    pb = PreprocessBase(train, test)
    pb.standard_scaler(['alcohol','ash'])

    print(pb.train_data)