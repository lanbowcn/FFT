from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import os
import csv

sess = tf.InteractiveSession()  # 创建session


# 一，函数声明部分

def weight_variable(shape):
    # 正态分布，标准差为0.1，默认最大为1，最小为-1，均值为0
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    # 创建一个结构为shape矩阵也可以说是数组shape声明其行列，初始化所有值为0.1
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # 卷积遍历各方向步数为1，SAME：边缘外自动补0，遍历相乘
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # 池化卷积结果（conv2d）池化层采用kernel大小为2*2，步数也为2，周围补0，取最大值。数据量缩小了4倍
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


# 二，定义输入输出结构

# 声明一个占位符，None表示输入图片的数量不定，101*99图片分辨率
xs = tf.placeholder(tf.float32, [None, 101, 99, 1])
# 类别是0-1总共2个类别，对应输出分类结果
ys = tf.placeholder(tf.float32, [None, 1])
keep_prob = tf.placeholder(tf.float32)
# x_image又把xs reshape成了101*99*1的形状，因为是灰色图片，所以通道是1.作为训练时的input，-1代表图片数量不定
x_image = tf.reshape(xs, [-1, 101, 99, 1])

# 三，搭建网络,定义算法公式，也就是forward时的计算

## 第一层卷积操作 ##
# 第一二参数值得卷积核尺寸大小，即patch，第三个参数是图像通道数，第四个参数是卷积核的数目，代表会出现多少个卷积特征图像;
W_conv1 = weight_variable([5, 5, 1, 64])
# 对于每一个卷积核都有一个对应的偏置量。
b_conv1 = bias_variable([64])
# 图片乘以卷积核，并加上偏执量，卷积结果16*15*64
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

##Flatten
#BatchNorm
h_norm2=tf.layers.batch_normalization(h_conv1,1)#待商榷
#2x2池化，边缘丢弃
h_pool2 = max_pool_2x2(h_norm2)#此时输出为7*8*64
# 扁平处理
h_pool2_flat = tf.reshape(h_pool2,[-1, 8*7*64])

##Dense全连接
W_fc1 = weight_variable([8*7*64,64])
b_fc1 = bias_variable([64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 对卷积结果执行dropout操作

##输出层
W_fc2 = weight_variable([64, 1])
b_fc2 = bias_variable([1])
# 最后的分类，结果为1*1*10 softmax和sigmoid都是基于logistic分类算法，一个是多分类一个是二分类
y_conv = tf.nn.sigmoid(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# 四，定义loss(最小误差概率)，选定优化优化loss，
cross_entropy = -tf.reduce_sum(ys * tf.log(y_conv))  # 定义交叉熵为loss函数
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)  # 调用优化器优化，其实就是通过喂数据争取cross_entropy最小化

# 五，开始数据训练以及评测

def readdata(batchnumber,batchsize):
    # 文件操作：测试提取一定规模的训练样本
    data=[]
    label=[]
    filepath = r'D:\xinyin\xinyin\training\training-a\spect'
    for (root, dir, files) in os.walk(filepath):
        # 加载 录音-标签 字典
        with open(os.path.join(r'D:\xinyin\xinyin\training\training-a\REFERENCE.csv'), 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]
            labedict = dict(rows)
            for file in files[(batchnumber-1)*batchsize:batchnumber*batchsize]:
                matrixpath = os.path.join(root, file)
                filename = file.split('-', 2)[0]
                my_matrix = np.loadtxt(open(matrixpath, "rb"), delimiter=",", skiprows=0)
                # print(os.path.join(root,file))输出带根目录的文件路径
                data.append(my_matrix)
                label.append(labedict[filename])
    return data,label

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


tf.global_variables_initializer().run()
ITERS = 20000#总迭代次数
BATCH_SIZE = 64

#剩余数据作为测试
test_data,test_label =readdata(41,BATCH_SIZE)

for i in range(1,40):
    data,label= readdata(i,64)#次数出现为list类型
    data = np.reshape(data, (64, 101, 99, 1))
    print(label)
    label = np.reshape(label,(-1,1))
    train_step.run(feed_dict={xs: data, ys: label, keep_prob: 0.5})
    # data.rehsape((data.shape[0], 101, 99, 1))
    if i % 2 == 0:
        train_accuracy = accuracy.eval(feed_dict={xs: test_data, ys: test_label, keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
print("test accuracy %g" % accuracy.eval(feed_dict={xs: test_data, ys: test_label, keep_prob: 1.0}))
# sess.run(accuracy, feed_dict=)
