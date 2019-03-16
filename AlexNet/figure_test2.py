import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


path1 = open('/home/kd/PycharmProjects/log/AlexNet_Baseline_Node1_2018-12-29 21:19:31.csv','rb')
data1 = pd.read_csv(path1)
data1 = data1.dropna(axis=0, how="any")
timestamp = data1['index']
test_accuracy = data1['test_accuracy']
path2 = open('/home/kd/PycharmProjects/log/AlexNet_Aji_Node1_2018-12-31 13:20:44.csv','rb')
data2 = pd.read_csv(path2)
data2 = data2.dropna(axis=0, how="any")
timestamp1 = data2['index']
test_accuracy1 = data2['test_accuracy']
path3 = open('/home/kd/PycharmProjects/log/AlexNet_Var_Node1_2018-12-30 18:34:17.csv','rb')
data3 = pd.read_csv(path3)
data3 = data3.dropna(axis=0, how="any")
timestamp2 = data3['index']
test_accuracy2 = data3['test_accuracy']
path4 = open('/home/kd/PycharmProjects/log/AlexNet_Var_momentum_Node1_2019-01-13 16:11:04.csv','rb')
data4 = pd.read_csv(path4)
data4 = data4.dropna(axis=0, how="any")
timestamp3 = data4['index']
test_accuracy3 = data4['test_accuracy']
plt.xlabel("timestamp",fontsize=12)
plt.ylabel("test_accuracy",fontsize=12)
plt.plot(timestamp, test_accuracy,'r')
plt.plot(timestamp1,test_accuracy1,'g')
plt.plot(timestamp2, test_accuracy2, 'b')
plt.plot(timestamp3, test_accuracy3, 'y')
plt.rc('grid', linestyle="-", color='black')
plt.savefig("AlexNet1.png")
plt.grid(True)
plt.show()