import pandas as pd
import numpy as np

from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam
from preprocessing import preprocess_data

cols_to_fe = [
    'EngineVersion', 'AppVersion', 'AvSigVersion', 'Census_OSVersion'
]

cols_to_ohe = [
    'RtpStateBitfield', 'IsSxsPassiveMode', 'DefaultBrowsersIdentifier',
    'AVProductStatesIdentifier', 'AVProductsInstalled', 'AVProductsEnabled',
    'CountryIdentifier', 'CityIdentifier',
    'GeoNameIdentifier', 'LocaleEnglishNameIdentifier',
    'Processor', 'OsBuild', 'OsSuite',
    'SmartScreen', 'Census_MDC2FormFactor',
    'Census_OEMNameIdentifier', 
    'Census_ProcessorCoreCount',
    'Census_ProcessorModelIdentifier',
    'Census_PrimaryDiskTotalCapacity',
    'Census_PrimaryDiskTypeName',
    'Census_HasOpticalDiskDrive',
    'Census_TotalPhysicalRAM',
    'Census_ChassisTypeName',
    'Census_InternalPrimaryDiagonalDisplaySizeInInches',
    'Census_InternalPrimaryDisplayResolutionHorizontal',
    'Census_InternalPrimaryDisplayResolutionVertical',
    'Census_PowerPlatformRoleName',
    'Census_InternalBatteryType',
    'Census_InternalBatteryNumberOfCharges',
    'Census_OSEdition',
    'Census_OSInstallLanguageIdentifier',
    'Census_GenuineStateName',
    'Census_ActivationChannel',
    'Census_FirmwareManufacturerIdentifier',
    'Census_IsTouchEnabled',
    'Census_IsPenCapable',
    'Census_IsAlwaysOnAlwaysConnectedCapable',
    'Wdft_IsGamer', 'Wdft_RegionIdentifier'
]

X_train, X_val, Y_train, Y_val = preprocess_data('assets/train.csv', 0.5, cols_to_fe, cols_to_ohe, 0.005, 5, verbose=True)

num_cols = len(X_train.columns)
