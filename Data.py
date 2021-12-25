import pandas as pd

from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split


# URL (google drive) of the NSL KDD Dataset (In CSV format)
download_url = 'https://drive.google.com/uc?id=1dhVPtvCy_F4_qWb2kkaZc6VOqlxW3LVl'
# df = pd.read_csv(download_url)
df = pd.read_csv('data/nsl_kdd.csv')

# print(df.shape)
# print(df.head())

def bool_attack(x):
    if(x != "normal"):
        return "attack"
    else:
        return "normal"

def convert_bool(x):
    if(x == "attack"):
        return 1
    else:
        return 0

def preprocess(df_x, df_y):
    df_x = pd.get_dummies(df_x, columns = ["protocol_type","service","flag"])
    x_normalise = Normalizer().fit(df_x)
    df_x = x_normalise.transform(df_x)
    x_new = SelectFpr(chi2, alpha=0.05).fit_transform(df_x, df_y)
    return x_new

df_x = df.drop('class', axis=1).drop('difficulty_level', axis=1)
df_y = df['class'].apply(bool_attack).apply(convert_bool)

x_new = preprocess(df_x, df_y)

_, n_columns = x_new.shape
print('Number of columns in preprocessed dataset: ' + str(n_columns))

x1, x2, y1, y2 = train_test_split(x_new, df_y, test_size=0.5, random_state=42)

# print(x1.shape)
# print(y1.shape)
# print(x2.shape)
# print(y2.shape)

# Exports: x1, y1, x2, y2