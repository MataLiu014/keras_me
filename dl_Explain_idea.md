
关于keras coding和我已有的想法
以下的全部#之后都是注释，注释好

关于python和R的代码的问题
其实感觉keras在python和R里面区别很小，基本的代码结构都是一样的

我现在的网络：


```python
import pandas as pd    #pandas是python里面的大数据处理库
import numpy as np      #numpy是python里面的常用科学计算库
import matplotlib.pyplot as plt     ##matplotlib.pyplot是一个画图工具
##以下的部分是加入keras需要的部分
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import Model
from keras.layers import Input
```


```python
#我这里用的是函数式模型，这个模型是老师ppt chap7 p22那种的，主要是可以包含更多的内容，对复杂的东西比sequential强

#这里我定义了一个子网络，用来提取过去24小时气温的一定特征，输入维度为24，输出维度为3
local=Input(shape=(24,))    #这是一个输入,对应ppt p22第一行
#之后对应p22 2-5行的东西
lax=Dense(30, activation="relu")(local)      #activation是激活函数，在chap6里面有讲
lax=Dense(20, activation="relu")(lax)        #python这里和R稍微有点区别，我需要在定义Dense()确定本层结构后再在后面加上上一层的变量名，R不需此操作
lax=Dense(10, activation="relu")(lax)
localfea=Dense(3 )(lax)

#这里我定义了一个子网络，用来提取过去7天气温的一定特征，输入维度为28（每天以lag=6输入4个数据点），输出维度为3
#结构和之前类似
global_=Input(shape=(28,))
glx=Dense(30, activation="relu")(global_)
glx=Dense(20, activation="relu")(glx)
glx=Dense(10, activation="relu")(glx)
globalfea=Dense(3 )(glx)

#我的两个子网络的数据需要被整合进最后的训练集，图解见老师ppt chap7 p21 这里的功能类似图中的add
#concatenate是merge层的一种，用来连接不同的张量（向量），e.g. concat([1,2],[3,4])=[1,2,3,4]
#老师结构里面是每个子网络提出了一个特征，其实可以理解成通过已有数据对温度的估计，add做的是相加后和结果比对，我觉得吧,emmm...
#我这里做的是把提出来的张量（向量）特征结合一下，再在后面的层最后进行一次提取
allfea=keras.layers.concatenate(inputs=[localfea,globalfea])  #这有点坑啊，[]和()在不同地方还不一样
alf=Dense(6,activation='elu')(allfea)
res=Dense(1)(alf)     #最后的输出

#用model把之前所有的东西结合起来
#model里面先用()将所有的输入变量名按顺序扩起来，然后再把所有的outputs扩起来
#对应p22最后一行
modelF = Model((local,global_), res)
```

关于编译模型和可选参数的含义:
python里面的complie的原型是这样的：
compile(self, optimizer, loss, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
这里我们主要要用的是optimizer和loss
self不用管，不用填，那个是面向对象编程的东西
optimizer是训练函数的优化器
我们知道深度神经网络用的是梯度下降法，但是具体怎么下降就是优化器的工作了，简单来说优化器决定了每个epoch训练中的loss的下降的效果和速度
好的优化器会大大提升训练的速度和效果，（并减少对电脑的摧残）
keras里面有两个比较先进的优化器adamax和nadam，它们比老师例子里面出现的rmsprop快得多也好得多
使用它们的方法：
adamax:modelF.compile(loss='MAE',optimizer='adamax')  #如果你的网络名名不叫modelF,那这个地方要写成你的网络名，比如说ppt上的例子这里是model_time，就应该写成model_time.compile(loss='MAE',optimizer='adamax') (python)
nadam:modelF.compile(loss='MAE',optimizer='nadam')
loss就是损失函数啦，这里估计也就用得上MAE和MSE了
我自己认为至少question1用MSE比MAE要好，因为预测温度和实际温度出现偏差的程度的实际影响不是线性的
比如说预测错了0.1摄氏度，那么生活体验不会有偏差，但是错了2摄氏度那有时体验会很不好
这里的影响MAE是0.1:2,MSE是0.01:4，我觉得MSE这个指标是更好的
metrics其实也可以制定accuracy作为一个判定网络优劣的标准，但是那个好像用在离散的分类问题上会好一点，我觉得在这课上意义不大
使用它们的方法：
MAE：modelF.compile(loss='MAE',optimizer='adamax')
MSE：modelF.compile(loss='MAE',optimizer='adamax')


```python
关于fit的解释：
python里面fit的原型是这样的：
fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
我们主要会用x,y,epochs和validation_data，其它的基本不用管
x就是你的训练输入
y是训练输出
validation_data是验证集
唔小伙伴们好像使用R的比较多，具体的编程细节原则我就不写了，其实还有点小坑
epoches就是过训练集的次数，这个需要自己指定
batch_size是batch的大小，其实keras会自动选择的，我是交给keras自动选择了，也不懂batch_size的具体影响
validation_split不要碰，那个东西输出的训练集验证集不符合时间序列的原则
```


```python
#最后的代码是这样的：
modelF.compile(loss='MAE',optimizer='adamax')
epoch=50
FHistory=modelF.fit(x=[x_trainG,x_trainH], y=y_trainG,epochs=epoch,validation_data=([x_testG,x_testH], y_testG))
#这里的一个细节：fit实际上会返回训练历史，我就存了下来，以便以后画图什么的
#只是训练的话modelF.fit(x=[x_trainG,x_trainH], y=y_trainG,epochs=epoch,validation_data=([x_testG,x_testH], y_testG))就好了
```

以上就是一个完整的网络了

Q1目前的思路：
我构造了两个输入：24小时的历史气温数据和lag=6hr的7天历史气温数据
我认为24小时历史代表气温的local trend
7天的历史代表总体气温气候的global trend
结合总体和局部进行预测

现在的效果（用MAE评估的）：
naive forcast: 我用的是用24小时前的气温估6hr之后的气温 MAE=0.26
现在的是 MAE=0.17
说实话这个结构效果并不好..我需要改改
