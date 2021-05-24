import joblib
import pandas as pd

from torch.utils.data import TensorDataset

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