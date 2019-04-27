from math import isnan
from numpy import where

from sklearn.model_selection import train_test_split


def is_nan(x):
    if isinstance(x, float):
        if isnan(x):
            return True

    return False


def frequency_encode(df, cols: [str]):
    for col in cols:
        d = df[col].value_counts(dropna=False)
        df[f'{col}_FE'] = df[col].map(d)/d.max()

        # print(f'Frequency encoded {col}')

    df.drop(columns=cols, inplace=True)


'''
Statistical One-Hot Encoding will disregard attributes with more categories than make sense to.
It detects this using a trick from statics in which you assume a random sample, and upon each value test the hypothesis:
    H0: Prob(p=1) == m
    HA: Prob(p=1) != m
where p is the observed target_col rate given value is present, and m is a value between 0 and 1.

Then Central Limit Theory tells us that:
    z == (p-m)/std_dev(p) == 2*(p-m)*(n//2)
where n is #occurrences of value

which is transformed below to determine whether or not to translate.
'''


def one_hot_encode(df, cols: [str], target_col: str = 'HasDetections',
                   filter: float = 0.005, z: float = 5, m: float = 0.5):
    for col in cols:
        value_counts = df[col].value_counts(dropna=False)

        for x, n in value_counts.items():
            if n < filter * len(df):
                break
            entriesWithValue = df[col].isna() if is_nan(x) else df[col] == x

            p = df[entriesWithValue][target_col].mean()

            if abs(p - m) > (z / n//2):
                df[f'{col}_BE_{x}'] = entriesWithValue.astype('int8')

        # print(f'OHEncoded {col} and created {len(value_counts)} flags')
    df.drop(columns=cols, inplace=True)


'''
Function to preprocess dataframe provided, uses test_size precentage of the data
in the train_test_split result, frequency encodes cols_to_fe, one-hot encodes cols_to_ohe,
passes ohe_filter, ohe_zvalue, and ohe_mval to the one_hot_encode function, uses target_column
as the label
'''


def preprocess_data(df, cols_to_fe: [str], cols_to_ohe: [str],
                    test_size: float = 0.3,
                    ohe_filter: float = 0.005, ohe_z: float = 5, ohe_m: float = 0.5,
                    target_col: str = 'HasDetections'):
    frequency_encode(df=df, cols=cols_to_fe)
    one_hot_encode(df=df, cols=cols_to_ohe, target_col=target_col, filter=ohe_filter, z=ohe_z, m=ohe_m)

    return train_test_split(df.drop(columns=[target_col]), df[target_col], test_size=test_size)
