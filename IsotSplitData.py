import pandas as pd
from sklearn.utils import shuffle

def get_isot_splits():
    
    s2 = pd.read_csv('data/isot/s2.csv', header = None)
    s3 = pd.read_csv('data/isot/s3.csv', header = None)
    s4 = pd.read_csv('data/isot/s4.csv', header = None)
    s5 = pd.read_csv('data/isot/s5.csv', header = None)
    s6 = pd.read_csv('data/isot/s6.csv', header = None)
    
    dataframes = [s2, s3, s4, s5, s6]
    splits = []
    
    for df in dataframes:
        df = shuffle(df)
        x = df.drop(208, axis=1)
        y = df[208]
        splits.append((x, y))
    
    return splits