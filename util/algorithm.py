# -*- coding: utf-8 -*-
"""
@author: Zhiwei Bao
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA


class Algorithm:

    def __init__(self, dataset):
        self._dataset_1 = dataset.iloc[:, :3]
        self._dataset_2 = dataset.iloc[:, 3:]

        self._train_data_1 = pd.DataFrame()
        self._train_data_2 = pd.DataFrame()
        self._test_data_1 = pd.DataFrame()
        self._test_data_2 = pd.DataFrame()

        self._q1 = np.array([])
        self._q2 = np.array([])
        self._threshold1 = 0
        self._threshold2 = 0

    def get_train_test(self):
        return {'train': [self._train_data_1, self._train_data_2], 'test': [self._test_data_1, self._test_data_2]}

    def get_q(self):
        return self._q1, self._q2

    def get_threshold(self):
        return self._threshold1, self._threshold2

    def form_train_test(self, split_point):
        self._train_data_1 = self._dataset_1.iloc[:split_point, :]
        self._train_data_2 = self._dataset_2.iloc[:split_point, :]
        self._test_data_1 = self._dataset_1.iloc[split_point:, :]
        self._test_data_2 = self._dataset_2.iloc[split_point:, :]

    def transform(self, trans_type):
        if trans_type == "Power":
            trans_scaler = PowerTransformer()
        else:
            trans_scaler = StandardScaler()

        trans_scaler.fit(self._train_data_1)
        self._train_data_1 = trans_scaler.transform(self._train_data_1)
        self._test_data_1 = trans_scaler.transform(self._test_data_1)

        trans_scaler.fit(self._train_data_2)
        self._train_data_2 = trans_scaler.transform(self._train_data_2)
        self._test_data_2 = trans_scaler.transform(self._test_data_2)

    def calculate_q(self):
        transformer = PCA(n_components=1, svd_solver='full')
        self._q1 = self.__calculate_q(transformer, self._train_data_1, self._test_data_1)
        self._q2 = self.__calculate_q(transformer, self._train_data_2, self._test_data_2)

    @staticmethod
    def __calculate_q(transformer, train_data, test_data):
        transformer.fit(train_data)
        pcs = transformer.transform(train_data)
        pcs_test = transformer.transform(test_data)
        train_data_est = transformer.inverse_transform(pcs)
        test_data_est = transformer.inverse_transform(pcs_test)
        res = train_data - train_data_est
        res_test = test_data - test_data_est
        q = np.array(res.dot(res.transpose())).diagonal()
        q_test = np.array(res_test.dot(res_test.transpose())).diagonal()
        return np.append(q, q_test)

    def calculate_threshold(self):
        self._threshold1 = self.__calculate_threshold(self._train_data_1)
        self._threshold2 = self.__calculate_threshold(self._train_data_2)

    @staticmethod
    def __calculate_threshold(data):
        V, D, Vt = np.linalg.svd(np.transpose(data).dot(data) / (len(data) - 1))
        theta1 = np.sum(D[1:] ** 1)
        theta2 = np.sum(D[1:] ** 2)
        theta3 = np.sum(D[1:] ** 3)
        h0 = 1 - 2 * theta1 * theta3 / (3 * theta2 ** 2)
        C_alpha = 2.33
        return theta1 * (C_alpha * h0 * np.sqrt(2 * theta2) / theta1 + 1 + theta2 * h0 * (h0 - 1) / theta1 ** 2) ** (
                1 / h0)
