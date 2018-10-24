
神经网络的调参：最最最最简单的说明
（求各位小伙伴有时间顺手一起调参...一个人搞这个太没效率了）


```python
##下面是我的比较完整的一个神经网络处理模型：
from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed
from keras.layers import LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
LAData=pd.read_csv("LA_temperature_normal.csv")
```


```python
def FormXTrainH(i):
    return [LATlist[i+j*3] for j in range(-29,-1)]      ##可调(1)

def FormYTrainH(i):
    return LATlist[i]

x_trainH=[]
x_testH=[]
y_testH=[]
y_trainH=[]

for i in range(29*3,32200+29*3):                    ##可调(1)
    #x_trainG.append( FormXTrainG(i)   )
    y_trainH.append( FormYTrainH(i)   )
    x_trainH.append(FormXTrainH(i))
for i in range(32200+29*3, len(LATlist)):           ##可调(1)
    #x_testG.append( FormXTrainG(i) )
    y_testH.append( FormYTrainH(i) )
    x_testH.append(FormXTrainH(i))
    
x_trainH=np.array(x_trainH)
x_testH=np.array(x_testH)
y_trainH=np.array(y_trainH)
y_testH=np.array(y_testH)
```


```python
epoch=50                        ##可调(4)

row_hidden=48                     ##可调(3)
col_hidden=48                     ##可调(3)
```


```python
x_trainH=x_trainH.reshape(x_trainH.shape[0],28,1)             ##可调(2)
x_testH=x_testH.reshape(x_testH.shape[0],28,1)
```


```python
row,col=x_trainH.shape[1:]               ##可调(2)

```


```python
x=Input(shape=(row,col))

#encoded rows
#encoded_rows = TimeDistributed(LSTM(row_hidden))(x)

#encoded cols
#encoded_cols= LSTM(col_hidden)(encoded_rows)
encoded_cols= LSTM(col_hidden)(x)

#最后的全连接
fin= Dense(10, activation='tanh')(encoded_cols)            ##可调(3)
#prediction= Dense(1, activation='tanh')(encoded_cols)

#
prediction=Dense(1,)(fin)                  

#summarizing and setting model
model = Model(x, prediction)
#这个model弄的很有趣，直接包含中间了的么 
model.compile(loss='MSE',                            ##可调(4)
              optimizer='adamax')                    ##可调(4)

model.summary()
```


```python
model.fit(x_trainH, y_trainH,
          batch_size=32,                           ##可调(4)
          epochs=epoch,
          verbose=1,
          validation_data=(x_testH, y_testH))
```

可以调整什么参数：
1. 输入数据的内容和构造
2. 输入数据的格式（比如说28*1也可以改成4*7）   关于这个格式的理解：keras文档里面的LSTM和Timedistributed层
3. 网络参数： LSTM的隐层(row_hidden,col_hidden), Dense的神经元数量和激活函数
4. 训练层次： loss函数，optimizer，epoch（网络下降的速度不一样，LSTM和全连接会有很大的差别，需要合适的epoch)，batch_size（也可以用不填的方式让网络自动选择epoches)


```python
上一段的LSTM：
LSTM是一种长短期记忆stateful RNN模型，比课上用到的Dense模型要复杂
LSTM非常适合时间序列的特征学习，however在Q1中没拿出突破性的表现（虽然也很不错）
```


```python
评估结果：
（1）training_loss 和 verification_loss 一起下降：
Epoch 1/50
32200/32200 [==============================] - 17s 513us/step - loss: 0.1992 - val_loss: 0.1076
Epoch 2/50
32200/32200 [==============================] - 14s 450us/step - loss: 0.1015 - val_loss: 0.0971
Epoch 3/50
32200/32200 [==============================] - 18s 570us/step - loss: 0.0951 - val_loss: 0.0911
Epoch 4/50
32200/32200 [==============================] - 19s 588us/step - loss: 0.0922 - val_loss: 0.0886
Epoch 5/50
32200/32200 [==============================] - 20s 632us/step - loss: 0.0898 - val_loss: 0.0844
这说明网络的表现在逐渐变好，这非常好

(2)training_loss 下降， verification_loss 稳定:
200/32200 [==============================] - 21s 666us/step - loss: 0.0748 - val_loss: 0.0773
Epoch 24/50
32200/32200 [==============================] - 21s 664us/step - loss: 0.0737 - val_loss: 0.0771
Epoch 25/50
32200/32200 [==============================] - 21s 664us/step - loss: 0.0728 - val_loss: 0.0774
Epoch 26/50
32200/32200 [==============================] - 23s 726us/step - loss: 0.0724 - val_loss: 0.0803
Epoch 27/50
32200/32200 [==============================] - 25s 774us/step - loss: 0.0718 - val_loss: 0.0802
Epoch 28/50
32200/32200 [==============================] - 24s 748us/step - loss: 0.0714 - val_loss: 0.0780
Epoch 29/50
32200/32200 [==============================] - 17s 516us/step - loss: 0.0708 - val_loss: 0.0834
网络并没有变好，并有过度拟合的倾向，但我觉得也不是没意义的

(3)training_loss 下降， verification_loss 上升:
32200/32200 [==============================] - 6s 193us/step - loss: 0.0376 - val_loss: 0.0919
Epoch 34/50
32200/32200 [==============================] - 6s 196us/step - loss: 0.0371 - val_loss: 0.0935
Epoch 35/50
32200/32200 [==============================] - 6s 197us/step - loss: 0.0365 - val_loss: 0.0960
Epoch 36/50
32200/32200 [==============================] - 6s 194us/step - loss: 0.0359 - val_loss: 0.0971
Epoch 37/50
32200/32200 [==============================] - 6s 195us/step - loss: 0.0355 - val_loss: 0.0955
Epoch 38/50
32200/32200 [==============================] - 6s 197us/step - loss: 0.0349 - val_loss: 0.0998
Epoch 39/50
32200/32200 [==============================] - 6s 196us/step - loss: 0.0341 - val_loss: 0.0996
Epoch 40/50
32200/32200 [==============================] - 6s 193us/step - loss: 0.0340 - val_loss: 0.1005
Epoch 41/50
32200/32200 [==============================] - 7s 207us/step - loss: 0.0336 - val_loss: 0.0986
Epoch 42/50
32200/32200 [==============================] - 8s 236us/step - loss: 0.0331 - val_loss: 0.1004
Epoch 43/50
32200/32200 [==============================] - 7s 231us/step - loss: 0.0325 - val_loss: 0.1011
过度拟合了，不是啥好现象....
```
