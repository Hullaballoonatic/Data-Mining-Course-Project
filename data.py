from os.path import isfile
import pandas as pd
import numpy as np

attributes = []


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
        tmp = [token.strip() for token in line.split(' - ', 2)]
        attributes.append(Attribute(tmp[0], tmp[1], tmp[2] if tmp[2] != 'NA' else None))


def get_data():
    train = pd.read_csv(train_raw_fp, index_col=0, dtype={attr.name: attr.dtype for attr in attributes})
    print(train)
    test = pd.read_csv(test_raw_fp, index_col=0, dtype={attr.name: attr.dtype for attr in attributes})
    print(test)

    train_label = train.HasDetections
    test_label = test.HasDetections

    return train.drop(columns='HasDetections'), train_label, test.drop(columns='HasDetections'), test_label
