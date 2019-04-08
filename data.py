from os.path import isfile
import pandas as pd
import numpy as np

attributes = {}


class Attribute:
    def __init__(self, name, dtype, description):
        self.name = name
        self.dtype = dtype
        self.description = description


train_fp = './assets/preprocessed/train.csv'
test_fp = './assets/preprocessed/test.csv'
train_raw_fp = './assets/raw/train.csv'
test_raw_fp = './assets/raw/test.csv'
data_description_fp = './assets/data.description.txt'

with open(data_description_fp, 'r') as file:
    for line in file.readlines():
        tmp = [token.strip() for token in line.split(' - ')]
        if len(tmp) > 3:
            tmp[2] = ' - '.join(tmp[2:])
        attributesToDTypeAndDescription[tmp[0]] = Attribute(tmp[0], tmp[1], tmp[2] if tmp[2] != 'NA' else None)


def get_data():
    if isfile(train_fp) and isfile(test_fp):
        print('preprocessed data exists.')
        train = pd.read_csv(train_fp, dtype=np.int8)
        test = pd.read_csv(test_fp, dtype=np.int8)
    else:
        print('preprocessed data does not exist. processing raw data.')
        train_raw = pd.read_csv(train_raw_fp, index_col=0)
        test_raw = pd.read_csv(test_raw_fp, index_col=0)

        print('transforming data into dummies')
        train = pd.get_dummies(train_raw)
        test = pd.get_dummies(test_raw)

        print('filling in missing data')
        trainAttrs = set(train)
        testAttrs = set(test)
        missing = trainAttrs - testAttrs
        if len(missing) > 0:
            for attr in missing:
                test[attr] = 0
        missing = testAttrs - trainAttrs
        if len(missing) > 0:
            for attr in missing:
                train[attr] = 0

        print('writing processed data to storage.')
        train.to_csv(train_fp)
        test.to_csv(test_fp)

    train_label = train.HasDetections
    test_label = test.HasDetections

    return train.drop(columns='HasDetections'), train_label, test.drop(columns='HasDetections'), test_label


train_samples, train_label, test_samples, test_label = get_data()
