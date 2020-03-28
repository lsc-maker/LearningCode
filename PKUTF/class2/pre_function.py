import tensorflow as tf

print("p4的代码：")
a = tf.constant([1, 2, 3, 1, 1])
b = tf.constant([0, 1, 3, 4, 5])
c = tf.where(tf.greater(a, b), a, b)  # 若a>b，返回a对应位置的元素，否则返回b对应位置的元素
print("若a>b，返回a对应位置的元素，否则返回b对应位置的元素：")
print("c：", c)

print("p5的代码：")
rdm = np.random.RandomState(seed=1)  #seed=常数每次生成随机数相同
d = rdm.rand()           #返回一个随机标量
e = rdm.rand(2, 3)       #返回一个维度为2行3列随机数矩阵
print("d:", d)
print("e:", e)

print("p6的代码：")
f = np.array([1, 2, 3])
g = np.array([4, 5, 6])
h = np.vstack((f, g))  #将两个数组按垂直方向叠加
print("将两个数组按垂直方向叠加：")
print("h:\n", h)

print("p7的代码：")
# 生成等间隔数值点
x, y = np.mgrid[1:3:1, 2:4:0.5]
# 将x, y拉直，并合并配对为二维张量，生成二维坐标点
grid = np.c_[x.ravel(), y.ravel()]
print("x:\n", x)
print("y:\n", y)
print("将x, y拉直，并合并配对为二维张量，生成二维坐标点：")
print("x.ravel():\n", x.ravel())
print("y.ravel():\n", y.ravel())
print('grid:\n', grid)
