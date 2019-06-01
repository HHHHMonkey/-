#coding:utf-8
'''
反向传播：训练模型参数，在所有参数上用梯度下降，使CNN模型在训练数据上的损失函数最小
损失函数loss:计算得到的预测值y与已知答案y_的差距，计算方法为交叉熵
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_lenet5_forward
import os
import numpy as np

#定义超参数：
BATCH_SIZE = 100#一次性喂入神经网络的数据大小
LEARNING_RATE_BASE =  0.005  #初始学习率衰减率
LEARNING_RATE_DECAY = 0.99  #衰减率
REGULARIZER = 0.0001 #正则化参数
STEPS = 50000 #训练轮数
MOVING_AVERAGE_DECAY = 0.99 
MODEL_SAVE_PATH="./model/" 
MODEL_NAME="mnist_model"  #模型保存的文件名

#反向传播，输入mnist
def backward(mnist):
    #输入x，y_进行占位
    #卷积层输入为四阶张量:第一阶表示每轮喂入的图片数量，第二阶和第三阶分别表示图片的行分辨率和列分辨率，第四阶表示通道数
    x = tf.placeholder(tf.float32,[BATCH_SIZE,#喂入图片数量
                                   mnist_lenet5_forward.IMAGE_SIZE,#行分辨率
                                   mnist_lenet5_forward.IMAGE_SIZE,#列分辨率
                                   mnist_lenet5_forward.NUM_CHANNELS]) #通道数
    y_ = tf.placeholder(tf.float32, [None, mnist_lenet5_forward.OUTPUT_NODE])
    
    '''首先进行前向传播'''
    y = mnist_lenet5_forward.forward(x,True, REGULARIZER)#调用前向传播算法，正则化参数为0.0001
    
    '''轮数计数器赋值0，设定为不可训练，即不会在训练的时候更新它的值'''
    global_step = tf.Variable(0, trainable=False) 
    
    '''定义损失函数，将softmax和交叉商协同使用，包含正则化'''
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))#把前向传播结果变成概率分布，再与训练集中的标准答案做对比，求出交叉熵
    cem = tf.reduce_mean(ce) #对向量求均值
    loss = cem + tf.add_n(tf.get_collection('losses')) #将参数w的正规化加入到总loss中，防止过拟合
    
    '''函数实现指数衰减学习率'''
    learning_rate = tf.train.exponential_decay( 
        LEARNING_RATE_BASE,#初始的学习率
        global_step,
        mnist.train.num_examples / BATCH_SIZE, #训练这么多轮之后开始衰减
		LEARNING_RATE_DECAY,#衰减指数
        staircase=True) 
    
    '''训练函数，使用随机梯度下降算法，使梯度沿着梯度的反方向（总损失减少的方向移动），实现参数更新'''
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    '''采用滑动平均的方法进行参数的更新'''
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]): 
        train_op = tf.no_op(name='train')

    '''保存模型'''
    saver = tf.train.Saver() #实例化saver对象

    '''训练过程'''
    with tf.Session() as sess: 
        #初始化所有参数
        init_op = tf.global_variables_initializer() 
        sess.run(init_op) 
        
        #断点续训 breakpoint_continue.py
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH) 
        if ckpt and ckpt.model_checkpoint_path:
        #恢复当前会话，讲ckpt中的值赋给w和b
        	saver.restore(sess, ckpt.model_checkpoint_path) 
          
        #循环迭代：    
        for i in range(STEPS):
            #讲训练集中batch_size的数据和标签赋给左边变量
            xs, ys = mnist.train.next_batch(BATCH_SIZE) 
            #修改喂入神经网络的参数
            reshaped_xs = np.reshape(xs,(  
		    BATCH_SIZE,
        	mnist_lenet5_forward.IMAGE_SIZE,
        	mnist_lenet5_forward.IMAGE_SIZE,
        	mnist_lenet5_forward.NUM_CHANNELS))
            #喂入神经网络，执行训练过程train_step
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys}) 
            if i % 100 == 0: 
                #每1000轮打印出当前的loss值
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                #循环1000轮保存模型到当前会话
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True) 
    #调用定义好的测试函数
    backward(mnist)
#判断python运行文件是否是为主文件，如果是，则执行
if __name__ == '__main__':
    main()


