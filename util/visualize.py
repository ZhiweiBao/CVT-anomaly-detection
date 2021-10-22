# -*- coding: utf-8 -*-
"""
@author: Zhiwei Bao
"""

import numpy as np
import matplotlib.pyplot as plt


class Visualization:

    def __init__(self, data1, data2, threshold1, threshold2):
        self.__window_size = 96
        self._plot_data_1 = self.__moving_average(data1, self.__window_size)
        self._plot_data_2 = self.__moving_average(data2, self.__window_size)
        self._plot_threshold_1 = threshold1
        self._plot_threshold_2 = threshold2

    def plot_anomaly_detecction_result(self):
        fig = plt.figure()
        plot_x = list(range(1, len(self._plot_data_1) + 1))

        # -----------------subplot1---------------------
        ax = fig.add_subplot(211)
        l1, = ax.plot(plot_x, self._plot_data_1, color='black', linewidth=1.0)
        cl1 = ax.hlines(self._plot_threshold_1, 1, len(plot_x) + 1, linestyles="dashed", color='red', linewidth=1.0)
        ax.vlines(2000, -1, 1, linestyles="dotted", color='black', linewidth=1.0)
        ax.set_xlim(0, 4512)
        ax.set_ylim(0, 0.225)
        ax.legend(handles=[l1, cl1], labels=['Q statistic', 'control limit'], loc='upper right')
        ax.set_ylabel("1st $Q$", fontsize=14)

        # -----------------subplot2---------------------
        ax = fig.add_subplot(212)
        l2, = ax.plot(plot_x, self._plot_data_2, color='black', linewidth=1.0)
        cl2 = ax.hlines(self._plot_threshold_1, 1, len(plot_x) + 1, linestyles="dashed", color='red', linewidth=1.0)
        ax.vlines(2000, -1, 1, linestyles="dotted", color='black', linewidth=1.0)
        ax.set_xlim(0, 4512)
        ax.set_ylim(0, 0.225)
        ax.legend(handles=[l2, cl2], labels=['Q statistic', 'control limit'], loc='upper right')
        ax.set_xlabel("Sample", fontsize=14)
        ax.set_ylabel("2nd $Q$", fontsize=14)
        fig.tight_layout()
        fig.savefig("three-phase dection.png", dpi=1200)

    def plot_fault_location_result(self):
        fig = plt.figure()
        plot_x = list(range(1, len(self._plot_data_1) + 1))
        ax = fig.add_subplot(111)
        l1, = ax.plot(plot_x, self._plot_data_1, linewidth=1.0)
        l2, = ax.plot(plot_x, self._plot_data_2, linewidth=1.0)
        ax.set_xlim(3000, 4512)
        ax.set_ylim(0, 0.175)
        ax.legend(handles=[l1, l2], labels=['1st group', '2nd group'], loc='best', fontsize=14)
        ax.set_xlabel("Sample", fontsize=14)
        ax.set_ylabel("$Q$", fontsize=14)
        fig.tight_layout()
        fig.savefig("three-phase location.png", dpi=1200)

    @staticmethod
    def __moving_average(data, window_size):
        window = np.ones(int(window_size)) / float(window_size)
        return np.convolve(data, window, 'same')

    @staticmethod
    def __smooth(a, WSZ):
        # a:原始数据，NumPy 1-D array containing the data to be smoothed
        # 必须是1-D的，如果不是，请使用 np.ravel()或者np.squeeze()转化
        # WSZ: smoothing window size needs, which must be odd number,
        # as in the original MATLAB implementation
        out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid') / WSZ
        r = np.arange(1, WSZ - 1, 2)
        start = np.cumsum(a[:WSZ - 1])[::2] / r
        stop = (np.cumsum(a[:-WSZ:-1])[::2] / r)[::-1]
        return np.concatenate((start, out0, stop))
