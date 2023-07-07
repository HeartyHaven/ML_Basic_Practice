# encoding=utf-8
import nu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import Linear_Regression as lr
from Linear_Regression import LinearRegression
# import csv
data=pd.read_csv('world-happiness-report.csv')
#获取训练和测试数据
train_data=data.sample(frac=0.8) #它的使用原理就是在序列a中随机抽取n个元素，并且将这n个元素按照以list的形式进行返回
"""
frac 设置抽样的比例，这里的意思是抽取80%的数据作为训练集；
random_state 是随机种子，可以随便写一个数字，起到了每次划分数据集都是按一样的规则的作用，这样每次运行随机函数的时候才能复现同样的结果；
"""
test_data=data.drop(train_data.index)
input_param_name='Log GDP per capita'
output_param_name='Positive affect'#这里是选择数据集中的一个属性来预测另外一个属性，只是演示，不要纠结实不实际
x_train=train_data[[input_param_name]].values
x_train[np.isnan(x_train)] = 0
"""
这里报错TypeError: 'numpy.float64' object does not support item assignment。
要不是跟着老师讲课，我恐怕又要花好几天解决这个问题了。因为我不熟悉列表的基本操作。
这个报错是因为前面的features_deviation是一个数字，而你试图从中找item，他就会报错float里面没有item因为是数字不是列表。
既然如此，肯定是读取数据的时候出现了问题。果不其然，一开始这些data里面只放了一个中括号。这就相当于是把所有的值都放出来，堆在一个大的数组里边。
解决方法是外面再套一层，这样的话，里面的每一个数字都成为了一个小列表，获取到的值经过一次列表变换再写入列表。这样就能够获取到item了，其实也就是他自己。
"""
y_train=train_data[[output_param_name]].values
y_train[np.isnan(y_train)] = 0
x_test=test_data[[input_param_name]].values
x_test[np.isnan(x_test)] = 0
y_test=test_data[[output_param_name]].values
y_test[np.isnan(y_test)] = 0#读取数据的时候记得处理nan值
plt.scatter(x_train,y_train,label='train_data')
plt.scatter(x_test,y_test,label='test_data')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
#这里就是尝试着把所有的点都显示一下子，看看长什么样
plt.title('happiness')
plt.legend()
plt.show()
#----------------------开始训练--------------------------------
num_iterations=500
learning_rate=0.01
linear_regression=LinearRegression(x_train,y_train)#初始化一个线性回归实例
(theta,costlist)=linear_regression.train(learning_rate,num_iterations)
print(costlist)
print('loss at first：',costlist[0])
print('loss after training：',costlist[-1])
plt.plot(range(num_iterations),costlist)
plt.xlabel('num_iter')
plt.ylabel('loss')
plt.title('GD')
plt.show()
predictions_num=100
x_predictions=np.linspace(x_train.min(),x_train.max(),100).reshape(-1,1)
print(x_predictions)
#这句话的意思是生成一个包含
# 100个元素的数组，该数组的值在x_train中最小值和最大值之间均匀分布。其中，np.linspace()函数用于生成等间距的数值序列。
y_predictions=linear_regression.predict(x_predictions)
plt.scatter(x_train,y_train,label='train_data')
plt.plot(x_predictions,y_predictions,'r',label='predictions')
plt.title('prediction conclusion')
plt.show()
#如果要绘制高级图片，可以参考plotly官网github，然后代码照抄，只更换数据就可以。
#另外，如果要使用notebook让图片嵌入显示，需要用iplot来做嵌入显示。