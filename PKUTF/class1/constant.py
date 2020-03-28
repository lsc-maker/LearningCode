import tensorflow as tf
import numpy as np

print("p17代码结果：")
print("创建一个张量：")
a = tf.constant([1,5],dtype=tf.int64) #创建一个张量
print(a)
print(a.dtype)
print(a.shape)

print("p18代码结果：")
c = np.arange(0,5)
d = tf.convert_to_tensor(a,dtype=tf.int64)  #将numpy数据类型转换为Tensorflow数据类型
print(c)
print("numpy数据类型转换为Tensorflow数据类型：")
print(d)

print("p19代码结果：")
e = tf.zeros([2,3])       #创建全为0的张量
f = tf.ones(4)           #创建全为1的张量
g = tf.fill([2,2],9)     #创建全为指定值的张量
print("创建全为0的张量：")
print(e)
print("创建全为1的张量：")
print(f)
print("创建全为指定值的张量：")
print(g)

print("p21代码结果：")
h = tf.random.normal([2,2],mean=0.5,stddev=1)    #生成正态分布的随机数，默认均值为0，标准差为1，(维度，mean=均值,stddev=标准差)
print("生成正态分布的随机数：")
print(h)
i = tf.random.truncated_normal([2,2],mean=0.5,stddev=1)  #生成截断式正态分布的随机数，默认均值为0，标准差为1，(维度，mean=均值,stddev=标准差)
print("生成截断式正态分布的随机数：")
print(i)

print("p22代码结果：")
j = tf.random.uniform([2,2],minval=0,maxval=1)   #生成均匀分布随机数(纬度，minval=最小值,maxval=最大值)
print("生成均匀分布随机数：")
print(j)
