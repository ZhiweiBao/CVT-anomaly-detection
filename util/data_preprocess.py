# -*- coding: utf-8 -*-
"""
@author: Zhiwei Bao
"""

import pandas as pd
import numpy as np


class DataPreprocess:

    def __init__(self):
        self._source_data = pd.DataFrame()
        self._dataset = pd.DataFrame()
        self._dataset_error = pd.DataFrame()
        self.__time_series = pd.DataFrame()

    def get_source_data(self):
        return self._source_data

    def get_dataset(self):
        return self._dataset

    def get_dataset_error(self):
        return self._dataset_error

    def data_import(self, data_dir1, data_dir2):
        source_data_1 = pd.read_excel(data_dir1, index_col=[0])
        source_data_2 = pd.read_excel(data_dir2, index_col=[0])
        self._source_data = pd.concat([source_data_1, source_data_2], axis=1)

    def preprocess(self, start, end, freq):
        self.__time_series = pd.date_range(start=start, end=end, freq=freq)
        dataset = self._source_data.loc[self.__time_series]
        dataset_1 = dataset.iloc[:, :3]
        dataset_2 = dataset.iloc[:, 3:]
        re_error = (dataset_1.values - dataset_2.values) / (dataset_1.values + dataset_2.values) * 2 * 100
        dataset.iloc[re_error[:, 0] < -.012] = np.nan
        dataset.iloc[re_error[:, 0] > .025] = np.nan
        dataset.iloc[re_error[:, 1] < -0.153] = np.nan
        dataset.iloc[re_error[:, 1] > -0.131] = np.nan
        # dataset.iloc[[682,2172,2174,3590]] = np.nan
        dataset.iloc[[1212, 1214, 2630]] = np.nan
        self._dataset = dataset.interpolate(method='linear')

    def increment_error(self, phase, error_type, error_point, error_factor):
        dataset_error = self._dataset.loc[self.__time_series]

        error_array = [1 for _ in range(error_point)]
        if error_type == "step error":
            error_array.extend([(1 + error_factor) for _ in range(len(dataset_error) - error_point)])
        elif error_type == "gradual error":
            error_array.extend([(1 + error_factor * i) for i in range(len(dataset_error) - error_point)])

        dataset_error.iloc[:, phase] = dataset_error.iloc[:, phase] * error_array
        self._dataset_error = dataset_error
