# -*- coding: utf-8 -*-
"""
Created on Fri May 31 19:59:04 2019

@author: 10670
"""
#coding:utf-8
#对于图片进行与处理函数
import sys; sys.path
import tensorflow as tf
import numpy as np
from PIL import Image
import mnist_lenet5_backward
import mnist_lenet5_forward

REGULARIZER = 0.0001 #正则化参数
BATCH_SIZE = 1 
#创建一个默认图，执行和train一样的操作
def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        #x = tf.placeholder(tf.float32,[None,mnist_lenet5_forward.INPUT_NODE])#只需要给x占位
        x = tf.placeholder(tf.float32,[BATCH_SIZE,#喂入图片数量
                               mnist_lenet5_forward.IMAGE_SIZE,#行分辨率
                               mnist_lenet5_forward.IMAGE_SIZE,#列分辨率
                               mnist_lenet5_forward.NUM_CHANNELS]) #通道数
        y = mnist_lenet5_forward.forward(x,True, REGULARIZER)#计算输出y
        preValue = tf.argmax(y,1)#y的最大值对应的列表索引号就是对以值
        
        '''实例化带有滑动平均值的saver'''
        variable_averages = tf.train.ExponentialMovingAverage(mnist_lenet5_backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        with tf.Session() as sess :
            '''断电续训'''
            ckpt = tf.train.get_checkpoint_state(mnist_lenet5_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                
                preValue = sess.run(preValue,feed_dict = {x:testPicArr})
                return preValue
            else:
                print("No checkponit file found")
                return -1
        
'''图片预处理操作'''      
def pre_pic(picName):
    img = Image.open(picName)#打开传入的原始图片
    reIm = img.resize((28,28),Image.ANTIALIAS)#用消除锯齿的方法将图片处理成28*28
    im_arr = np.array(reIm.convert('L'))#将reIm以灰度的形式转换成矩阵
    '''模型要求的是黑底白字，我们输入的是白底黑字,要给输入图片反色'''
    threshold=50
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255-im_arr[i][j]#每个像素点的新值，为互补的反色
            if (im_arr[i][j]<threshold):#二值化处理滤掉噪声
                im_arr[i][j] = 0 #小于阈值的点认为为纯黑色0
            else:im_arr[i][j] = 255 #大于阈值的点认为为纯白色255

    nm_arr = im_arr.reshape([1,28,28,1]) #整理形状，起名为nm_arr
    nm_arr = nm_arr.astype(np.float32) #模型要求的像素点为0~1的浮点数
    img_ready = np.multiply(nm_arr,1.0/255.0)
    return img_ready  #        
        
def application(path = None):
    if path == None:     #从命令行输入
        testNum = int(input("input the number of test picture:"))#要输入图片的数量
        for i in range(testNum):
            testPic =input("the path of test picture:")#给出图片的路径和名词
            '''将图片转换成能够输入神经网络的值'''
            #print(type(testPic))
            testPicArr = pre_pic(testPic)#对手写数字图片进行预处理
            '''将处过后的图片输入神经网络'''
            preValue = restore_model(testPicArr)
            print ("The prediction number is:",preValue) #输出预测值 
    else:    #从GUI输入
        testPicArr = pre_pic(path)#对手写数字图片进行预处理
        return restore_model(testPicArr)
          
        
def main():
    application()
    
if __name__ == '__main__':
    main()        