import pandas as pd
import numpy as np
import glob

from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from MyUtils import bool_attack, convert_bool
from Args import args

# URL (google drive) of the NSL KDD Dataset (In CSV format)
# download_url = 'https://drive.google.com/uc?id=1dhVPtvCy_F4_qWb2kkaZc6VOqlxW3LVl'
# df = pd.read_csv(download_url)


def preprocess(df_x, df_y):
    df_x = pd.get_dummies(df_x, columns = ["protocol_type","service","flag"])
    x_normalise = Normalizer().fit(df_x)
    df_x = x_normalise.transform(df_x)
    x_new = SelectFpr(chi2, alpha=0.05).fit_transform(df_x, df_y)
    return x_new

    
def get_nsl_random_splits():
    df = pd.read_csv('data/nsl/nsl_kdd.csv')
    df = shuffle(df, random_state=args.random_state)
    
    df_x = df.drop('class', axis=1).drop('difficulty_level', axis=1)
    df_y = df['class'].apply(bool_attack).apply(convert_bool)
    x_new = preprocess(df_x, df_y)
    
    _, n_columns = x_new.shape
    print('Number of columns in preprocessed dataset: ' + str(n_columns))
    
    xs = np.array_split(x_new, args.num_clients)
    ys = np.array_split(df_y, args.num_clients)
    
    splits = []
    for x, y in zip(xs, ys):
        splits.append((x, y))
        
    return splits


# =============================================================================
# def get_isot_random_splits():
#     df = pd.read_csv('data/isot/overall.csv', header = None)
#     df = shuffle(df, random_state=args.random_state)
#     
#     x = df.drop(209, axis=1)
#     y = df[209]
#     
#     xs = np.array_split(x, args.num_clients)
#     ys = np.array_split(y, args.num_clients)
#     
#     splits = []
#     for x, y in zip(xs, ys):
#         splits.append((x, y))
#         
#     return splits
# =============================================================================

def get_isot_random_splits():
    
    files = glob.glob('data/isot/overall_normalized/*.csv')
    lis = []
    
    for f in files:
        df = pd.read_csv(f)
        lis.append(df)

    df = pd.concat(lis)
    df = shuffle(df)

    x = df.drop('class', axis=1)
    y = df['class']

    xs = np.array_split(x, args.num_clients)
    ys = np.array_split(y, args.num_clients)

    splits = []
    for x, y in zip(xs, ys):
        splits.append((x, y))

    return splits
    
