# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 14:07:11 2020

@author: Administrator
"""


# -*- coding: utf-8 -*-

"""

 LSTM prediction

@author: ljq



"""

#导入库函数

import numpy 
import pandas as pd
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

df = pd.read_excel('C:\\Users\\Administrator\\Desktop\\TimeSeries\\4567均值.xlsx')
df = df.values.reshape(-1,1) #将一维数组，转化为多维数组

def create_dataset(dataset, look_back=1): #后一个数据和前look_back个数据有关系
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a) 
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY) #生成输入数据和输出数据 

scaler = MinMaxScaler(feature_range=(0, 1)) #归一化0-1
df = scaler.fit_transform(df)

#划分训练集测试集
train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
train, test = df[0:train_size,:], df[train_size:len(df),:] #训练集和测试集
look_back = 10
trainX , trainY = create_dataset(train, look_back) #训练输入输出
testX, testY = create_dataset(test, look_back) #测试输入输出
#reshape input to be [samples, time steps, features] 
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#建立LSTM模型
model = Sequential()
model.add(LSTM(11, input_shape=(1, look_back))) #隐层11个神经元 （可以调整此参数提高预测精度）
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam') #评价函数mse，优化器adam
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2) #100次迭代 
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

#数据反归一化
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
trainPredictPlot = numpy.empty_like(df)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

 

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(df)-1, :] = testPredict

# plot baseline and predictions
plt.figure(figsize=(20,6))
l1,=plt.plot(scaler.inverse_transform(df),color='red',linewidth=5,linestyle='--')
l2,=plt.plot(trainPredictPlot,color='k',linewidth=4.5)
l3,=plt.plot(testPredictPlot,color='g',linewidth=4.5)
plt.legend([l1,l2,l3],('raw-data','true-values','pre-values'),loc='best')
plt.show()
