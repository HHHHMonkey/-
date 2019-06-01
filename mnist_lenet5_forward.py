#coding:utf-8
import tensorflow as tf
IMAGE_SIZE = 28
NUM_CHANNELS = 1    # 因为是灰度图像，所以通道数为1
CONV1_SIZE = 5
CONV1_KERNEL_NUM = 32
CONV2_SIZE = 5
CONV2_KERNEL_NUM = 64
FC_SIZE = 512
OUTPUT_NODE = 10

def get_weight(shape, regularizer):
	w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))     # 从截断的正态分布中输出随机值,shape表示生成张量的维度,stddev是标准差
    # 如果正则化，则把“tf.contrib......(w)”添加到列表losses中
    # “tf.contrib......(w)”代表使用L2正则化规则
	if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w)) 
	return w

def get_bias(shape): 
	b = tf.Variable(tf.zeros(shape))  # 权重初始化为0
	return b

# 计算卷积
# x：输入描述->[batch,分辨率*分辨率，通道数]
# w：卷积核描述->[行分辨率，列分辨率，通道数，卷积核个数]
# strides：核滑动步长->[1,行步长，列步长，1]
# padding：填充，'SAME'表示用0填充（填充方式见CSDN中的CNN简介
def conv2d(x,w):    
	return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

# 采用最大池化
# x：输入描述->[batch,分辨率*分辨率，通道数]
# ksize：池化核描述->[1,行分辨率，列分辨率，1]
# strides：池化核滑动步长->[1,行步长，列步长，1]
# padding：填充，'SAME'表示用0填充（填充方式见CSDN中的CNN简介
def max_pool_2x2(x):  
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') 

def forward(x, train, regularizer):
    conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM], regularizer) 
    conv1_b = get_bias([CONV1_KERNEL_NUM]) 
    conv1 = conv2d(x, conv1_w) 
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b)) 
    pool1 = max_pool_2x2(relu1) 

    conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM],regularizer) 
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv2 = conv2d(pool1, conv2_w) 
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2 = max_pool_2x2(relu2) # 第二个池化层的输出

    pool_shape = pool2.get_shape().as_list() 
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3] 
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes]) # 将pool2的形状转化为pool_shape[0]行，nodes列

    fc1_w = get_weight([nodes, FC_SIZE], regularizer)   # fc1_w为nodes行，FC_SIZE列的矩阵
    fc1_b = get_bias([FC_SIZE]) 
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b) # FC全连接层的输出
    if train: fc1 = tf.nn.dropout(fc1, 0.5) #为了防止输出过拟合，x->输入tensor；0.5->每个神经元被留下的概率

    fc2_w = get_weight([FC_SIZE, OUTPUT_NODE], regularizer)
    fc2_b = get_bias([OUTPUT_NODE])
    y = tf.matmul(fc1, fc2_w) + fc2_b
    return y 
