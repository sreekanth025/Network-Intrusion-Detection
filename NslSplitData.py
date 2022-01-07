import pandas as pd

from MyUtils import convert_bool, bool_attack 

def get_nsl_splits():
    
    set_1 = pd.read_csv('data/nsl-splits/set-1.csv')
    set_2 = pd.read_csv('data/nsl-splits/set-2.csv')
    set_3 = pd.read_csv('data/nsl-splits/set-3.csv')
    set_4 = pd.read_csv('data/nsl-splits/set-4.csv')
    
    set_3 = pd.concat([set_3, set_4])
    
    dataframes = [set_1, set_2, set_3]
    splits = []
    
    for df in dataframes:
        x = df.drop('class', axis=1)
        y = df['class'].apply(bool_attack).apply(convert_bool)
        splits.append((x, y))
    
    return splits