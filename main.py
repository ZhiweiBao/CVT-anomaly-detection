# -*- coding: utf-8 -*-
"""
@author: Zhiwei Bao
"""

from util.data_preprocess import DataPreprocess
from util.algorithm import Algorithm
from util.visualize import Visualization
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # ------------------preprocessing--------------------
    dp = DataPreprocess()
    dp.data_import("./data/Source_data_1.xlsx", "./data/Source_data_2.xlsx")
    dp.preprocess(start='2018-02-01 00:00:00', end='2018-03-19 23:59:59', freq='15min')
    dp.increment_error(1, "step error", 3500, 0e-3)
    dataset = dp.get_dataset()

    # ------------------analysis--------------------
    alg = Algorithm(dataset)
    alg.form_train_test(2000)
    alg.transform("Standard")
    alg.calculate_q()
    alg.calculate_threshold()
    q1, q2 = alg.get_q()
    q_sigma_1, q_sigma_2 = alg.get_threshold()

    # ------------------visualization--------------------
    visual = Visualization(q1, q2, q_sigma_1, q_sigma_2)
    visual.plot_anomaly_detecction_result()
    visual.plot_fault_location_result()
    plt.show()
