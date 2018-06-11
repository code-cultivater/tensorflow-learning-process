import input_data;
import tensorflow as tf;
import numpy as np;
#inport data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True);

#graph

# input
x=tf.placeholder(tf.float32,[None,784]);

#cal
#W=tf.Variable(tf.zeros([784,10]));
W=tf.Variable(np.random.rand(784,10),dtype=tf.float32);
b=tf.Variable(tf.zeros(10),dtype=tf.float32);
b=tf.reshape(tf.tile(b,[tf.shape(x)[0]]),[-1,10]);

# res
y=tf.nn.softmax(tf.matmul(x,W)+b);

y_=tf.placeholder(tf.float32,[None,10]);

#define loss
loss=tf.reduce_sum(-tf.multiply(y_,tf.log(y)));

#optimizer
optimizer=tf.train.GradientDescentOptimizer(0.01).minimize(loss);

#init
init=tf.global_variables_initializer();

#sess
with tf.Session() as sess:
    sess.run(init);

    for i in range(1000):
        dx,dy=mnist.train.next_batch(100);
        sess.run(optimizer,feed_dict={x:dx,y_:dy});

    # acurrancy cal
    correct_rate = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32));
    rate_i = sess.run(correct_rate, feed_dict={x: mnist.test.images,y_:mnist.test.labels});
    print("correct rate : ",rate_i);

