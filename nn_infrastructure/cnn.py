import tensorflow as tf;

def conv2d(x,w,strides):
    return tf.nn.conv2d(x,w,strides=strides,padding="SAME");

def max_pool(x,ksize,strides):
    return tf.nn.max_pool(x,ksize=ksize,strides=strides,padding="SAME");

