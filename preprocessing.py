import math
import pandas as pd
import gc

from sklearn.model_selection import train_test_split

def is_nan(x):
        if isinstance(x, float):
            if math.isnan(x):
                return True

        return False

def frequency_encode(dataframe, column, verbose=False):
    d = dataframe[column].value_counts(dropna=False)
    n = column + "_FE"
    dataframe[n] = dataframe[column].map(d)/d.max()

    if (verbose):
        print(f'Frequency encoded {column}')

    return [n]

'''
Encode categorical attributes that comprise more than filter percent of the total data
and have a significance greater than the given z_value.
'''
def one_hot_encode(dataframe, column, filter, z_value, target_column='HasDetections', m=0.5, verbose=False):
    cv = dataframe[column].value_counts(dropna=False)
    cvd = cv.to_dict()
    vals = len(cv)
    th = filter * len(dataframe)
    sd = z_value * 0.5 / math.sqrt(th)

    n = []
    ct = 0
    d = {}

    for x in cv.index:
        try:
            if cv[x] < th:
                break
            
            sd = z_value * 0.5 / math.sqrt(cv[x])
        except:
            if cvd[x] < th:
                break

            sd = z_value * 0.5 / math.sqrt(cvd[x])

        if is_nan(x):
            r = dataframe[dataframe[column].isna()][target_column].mean()
        else:
            r = dataframe[dataframe[column] == x][target_column].mean()

        if abs(r - m) > sd:
            nm = column + '_BE_' + str(x)

            if is_nan(x):
                dataframe[nm] = (dataframe[column].isna()).astype('int8')
            else:
                dataframe[nm] = (dataframe[column] == x).astype('int8')
            
            n.append(nm)
            d[x] = 1
        
        ct += 1

        if (ct + 1) >= vals:
            break
    
    if (verbose):
        print(f'OHE encoded {column} and created {len(d)} booleans')

    return [n, d]

'''
Function to preprocess data. Loads data from csv_path, uses test_size precentage of the data
in the train_test_split result, frequency encodes cols_to_fe, one-hot encodes cols_to_ohe,
passes ohe_filter, ohe_zvalue, and ohe_mval to the one_hot_encode function, uses target_column
as the label, uses sample_size as the number of records to use from the loaded csv, and uses the
verbose boolean to determine whether or not to print verbose output.
'''
def preprocess_data(csv_path, test_size, cols_to_fe, cols_to_ohe, ohe_filter, ohe_zvalue, ohe_mval=0.5,
    target_column='HasDetections', sample_size=-1, verbose=False):

    dtypes = {}

    for x in cols_to_ohe + cols_to_fe:
        dtypes[x] = 'category'
    
    dtypes['MachineIdentifier'] = 'str'
    dtypes['HasDetections'] = 'int8'

    df_train = pd.read_csv(csv_path, usecols=dtypes.keys(), dtype=dtypes)

    if (verbose):
        print(f'Loaded {len(df_train)} rows from {csv_path}')

    if (sample_size != -1):
        df_train = df_train.sample(sample_size)

        if (verbose):
            print(f'Sample size of {sample_size} rows being used from {csv_path}')

    x = gc.collect()

    cols = []
    dd = []

    for x in cols_to_fe:
        cols += frequency_encode(df_train, x, verbose)
    
    for x in cols_to_ohe:
        tmp = one_hot_encode(df_train, x, ohe_filter, ohe_zvalue, target_column, ohe_mval, verbose)
        cols += tmp[0]
        dd.append(tmp[1])
    
    for x in cols_to_fe + cols_to_ohe:
        del df_train[x]
    
    if (verbose):
        print(f'Removed original {len(cols_to_fe + cols_to_ohe)} variables')

    return train_test_split(df_train[cols], df_train[target_column], test_size=test_size)
