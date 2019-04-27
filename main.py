import pandas as pd
import numpy as np

from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam
from preprocessing import preprocess_data

cols_to_fe = ['EngineVersion', 'AppVersion', 'AvSigVersion', 'Census_OSVersion']

cols_to_ohe = [
    'RtpStateBitfield', 'DefaultBrowsersIdentifier',
    'AVProductStatesIdentifier', 'AVProductsInstalled', 'AVProductsEnabled',
    'CountryIdentifier', 'CityIdentifier', 'GeoNameIdentifier',
    'LocaleEnglishNameIdentifier', 'Processor', 'OsBuild', 'OsSuite',
    'SmartScreen', 'Census_MDC2FormFactor', 'Census_OEMNameIdentifier',
    'Census_ProcessorCoreCount', 'Census_ProcessorModelIdentifier',
    'Census_PrimaryDiskTotalCapacity', 'Census_PrimaryDiskTypeName',
    'Census_HasOpticalDiskDrive', 'Census_TotalPhysicalRAM',
    'Census_ChassisTypeName', 'Census_InternalPrimaryDiagonalDisplaySizeInInches',
    'Census_InternalPrimaryDisplayResolutionHorizontal',
    'Census_InternalPrimaryDisplayResolutionVertical',
    'Census_PowerPlatformRoleName', 'Census_InternalBatteryType',
    'Census_InternalBatteryNumberOfCharges', 'Census_OSEdition',
    'Census_OSInstallLanguageIdentifier', 'Census_GenuineStateName',
    'Census_ActivationChannel', 'Census_FirmwareManufacturerIdentifier',
    'Census_IsTouchEnabled', 'Census_IsPenCapable',
    'Census_IsAlwaysOnAlwaysConnectedCapable', 'Wdft_IsGamer',
    'Wdft_RegionIdentifier'
]

dtype = {
    'MachineIdentifier': 'object',
    'EngineVersion': 'category',
    'AppVersion': 'category',
    'AvSigVersion': 'category',
    'Census_OSVersion': 'category',
    'RtpStateBitfield': 'category',
    'IsSxsPassiveMode': 'int8',
    'DefaultBrowsersIdentifier': 'category',
    'AVProductStatesIdentifier': 'category',
    'AVProductsInstalled': 'category',
    'AVProductsEnabled': 'category',
    'CountryIdentifier': 'category',
    'CityIdentifier': 'category',
    'GeoNameIdentifier': 'category',
    'LocaleEnglishNameIdentifier': 'category',
    'Processor': 'category',
    'OsBuild': 'category',
    'OsSuite': 'category',
    'SmartScreen': 'category',
    'Census_MDC2FormFactor': 'category',
    'Census_OEMNameIdentifier': 'category',
    'Census_ProcessorCoreCount': 'category',
    'Census_ProcessorModelIdentifier': 'category',
    'Census_PrimaryDiskTotalCapacity': 'category',
    'Census_PrimaryDiskTypeName': 'category',
    'Census_HasOpticalDiskDrive': 'category',
    'Census_TotalPhysicalRAM': 'category',
    'Census_ChassisTypeName': 'category',
    'Census_InternalPrimaryDiagonalDisplaySizeInInches': 'category',
    'Census_InternalPrimaryDisplayResolutionHorizontal': 'category',
    'Census_InternalPrimaryDisplayResolutionVertical': 'category',
    'Census_PowerPlatformRoleName': 'category',
    'Census_InternalBatteryType': 'category',
    'Census_InternalBatteryNumberOfCharges': 'category',
    'Census_OSEdition': 'category',
    'Census_OSInstallLanguageIdentifier': 'category',
    'Census_GenuineStateName': 'category',
    'Census_ActivationChannel': 'category',
    'Census_FirmwareManufacturerIdentifier': 'category',
    'Census_IsTouchEnabled': 'int8',
    'Census_IsPenCapable': 'int8',
    'Census_IsPortableOperatingSystem': 'int8',
    'Census_IsSecureBootEnabled': 'int8',
    'Census_IsAlwaysOnAlwaysConnectedCapable': 'category',
    'Wdft_IsGamer': 'category',
    'Wdft_RegionIdentifier': 'category',
    'HasTpm': 'int8',
    'IsBeta': 'int8',
    'HasDetections': 'int8'
}

df = pd.read_csv('assets/raw/train.csv', index_col=0, dtype=dtype)  # add nrows=n value for smaller sample size.
train, train_label, test, test_label = preprocess_data(df, cols_to_fe, cols_to_ohe)
