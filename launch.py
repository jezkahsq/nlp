# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 09:28:48 2020

@author: Administrator
"""


import pandas as pd
import matplotlib.pylab as plt
from TimeSeries.time_series_model import Model
import warnings
warnings.filterwarnings("ignore") # specify to ignore warning messages

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] =False

period = 95
p_max = 2
q_max = 1
d_max = 2
forecast_num = 300

df = pd.read_excel('C:\\Users\\Administrator\\Desktop\\TimeSeries\\时序.xlsx')

My_model = Model(df, period, p_max, q_max, d_max, forecast_num)
My_model.diff_process()
My_model.acf_pacf_fig()
My_model.params_select()
My_model.sarima()
My_model.model_eval()
My_model.qq_plot()