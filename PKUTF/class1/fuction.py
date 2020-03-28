import tensorflow as tf
import numpy as np

print("p23代码结果：")
x1 = tf.constant([1.,2.,3.],dtype=tf.float64)
print(x1)
x2 = tf.cast(x1,tf.int32) #强制tensor转换为该数据类型
print("强制tensor转换为该数据类型：")
print(x2)
print("计算张量维度上元素的最大值和最小值：")
print(tf.reduce_min(x2),tf.reduce_max(x2)) #计算张量维度上元素的最大值和最小值

print("p25的代码结果：")
x3 = tf.constant([[1,2,3],[2,2,3]])
print(x3)
print("计算张量沿着指定维度的平均值：")
print(tf.reduce_mean(x3))         #计算张量沿着指定维度的平均值
print("计算张量沿着制定维度的和：")
print(tf.reduce_sum(x3,axis=1))   #计算张量沿着制定维度的和

print("p29的代码结果：")
a = tf.ones([1,3])
b = tf.fill([1,3],3.)
print(a)
print(b)
print("张量元素相加：")
print(tf.add(a,b))            #张量元素相加
print("张量元素相减：")
print(tf.subtract(a,b))      #张量元素相减
print("张量元素相乘：")
print(tf.multiply(a,b))      #张量元素相乘
print("张量元素相除：")
print(tf.divide(a,b))        #张量元素相除
print("只有维度相同的张量才可以做四则运算")

print("p30的代码结果：")
c = tf.fill([1,2],3.)
print(c)
print("张量的3次方：")
print(tf.pow(c,3))        #计算张量的N次方
print("张量的平方：")
print(tf.square(c))       #计算张量的平方
print("张量的开方：")
print(tf.sqrt(c))         #计算张量的开方

print("p31的代码结果")
d = tf.ones([3,2])
e = tf.fill([2,3],3.)
print("矩阵相乘：")
print(tf.matmul(d,e))  #矩阵相乘

print("p33的代码结果：")
features = tf.constant([12,23,10,17])
labels = tf.constant([0,1,1,0])
dataset = tf.data.Dataset.from_tensor_slices((features,labels)) #切分传入张量第一维度，生成传入特征/标签对，构建数据集 numpy也可以使用
print(dataset)
for element in dataset:
	print(element)

print("p34的代码结果：")
with tf.GradientTape()as tape:
	w = tf.Variable(tf.constant(3.0))
	loss = tf.pow(w,2)
grad = tape.gradient(loss,w)  #求张量梯度
print("张量的梯度：")
print(grad)

print("p35的代码结果：")
seq = ['one','two','three']
print("遍历每个元素：")
for i,element in enumerate(seq):  #遍历每个元素
	print(i,element)

print("p37的代码结果：")
classes = 3
labels = tf.constant([1,0,2])
output = tf.one_hot(labels,depth=classes)   #独热码
print(output)

print("p39的代码结果：")
y = tf.constant([1.01,2.01,-0.66])
y_pro = tf.nn.softmax(y)  #输出符合概率分布
print("After softmax,y_pro is:{}".format(y_pro))

print("p40的代码结果：")
z = tf.Variable(4)
z.assign_sub(1)   #更新参数值
print(z)

print("p41的代码结果：")
test = np.array([[1,2,3],[2,3,4],[5,4,3],[8,7,2]])
print(test)
print("返回每一列(经度)最大值的索引：")
print(tf.argmax(test,axis = 0))  #返回每一列(经度)最大值的索引
print("返回每一行(维度)最大值的索引：")
print(tf.argmax(test,axis = 1))  #返回每一行(维度)最大值的索引