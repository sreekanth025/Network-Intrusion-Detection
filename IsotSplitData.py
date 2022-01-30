import pandas as pd
import glob
from sklearn.utils import shuffle

def get_isot_splits():
    
    files = glob.glob('data/isot/day_wise_normalized/*.csv')    
    dataframes = []
    splits = []
    
    for file in files:
        df = pd.read_csv(file)
        dataframes.append(df)
    
    for df in dataframes:
        df = shuffle(df)
        x = df.drop('class', axis=1)
        y = df['class']
        splits.append((x, y))
    
    return splits