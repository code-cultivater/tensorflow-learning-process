import tensorflow as tf
import input_data;
import nn_infrastructure.initial_vaiable as nn_initial;
import nn_infrastructure.cnn as cnn;

#import data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True);

# in&out
x=tf.placeholder(tf.float32,[None,784]);
x_image=tf.reshape(x,[-1,28,28,1]);
y_=tf.placeholder(tf.float32,[None,10]);

#initial variable
##conv1
W1=nn_initial.get_noisy_variable([5,5,1,32]);
b1=nn_initial.get_constant_variable([32]);
##conv2
W2=nn_initial.get_noisy_variable([5,5,32,64]);
b2=nn_initial.get_constant_variable([64]);
##fnn 3
W3=nn_initial.get_noisy_variable([7*7*64,1024]);
b3=nn_initial.get_constant_variable([1024]);
##fnn4
W4=nn_initial.get_noisy_variable([1024,10]);
b4=nn_initial.get_constant_variable([10]);


#model
l1_conv=tf.nn.relu(cnn.conv2d(x_image,W1,[1,1,1,1])+b1);
l1_pool=cnn.max_pool(l1_conv,[1,2,2,1],[1,2,2,1]);

l2_conv=tf.nn.relu(cnn.conv2d(l1_pool,W2,[1,1,1,1])+b2);
l2_pool=cnn.max_pool(l2_conv,[1,2,2,1],[1,2,2,1]);

l2_reshape=tf.reshape(l2_pool,[-1,7*7*64]);

l3=tf.nn.relu(tf.matmul(l2_reshape,W3)+b3);

    ##define dropout
keep_rate=tf.placeholder(tf.float32);
l3_drop=tf.nn.dropout(l3,keep_rate);

l4=tf.nn.softmax(tf.matmul(l3_drop,W4)+b4);

#optimizer
cross_entropy=tf.reduce_sum(tf.multiply(y_,tf.log(l4)));
optimizer=tf.train.AdamOptimizer(0.0001).minimize(cross_entropy);

#accuracy


#sess
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer());
    #train
    local_error_results_mean=0;
    for i in range(20000):
        data=mnist.train.next_batch(50);
        sess.run(optimizer,feed_dict={x:data[0],y_:data[1],keep_rate:0.5});
        local_error=tf.subtract(tf.constant(1.0),tf.reduce_mean(tf.cast(tf.equal(tf.argmax(l4,axis=1),tf.argmax(y_,1)),dtype=tf.float32)));
        local_error_result=sess.run(local_error,feed_dict={x:data[0],y_:data[1],keep_rate:1});
        local_error_results_mean+=local_error_result;
        print("round %d : %f"%(i,local_error_results_mean/(i+1)));
    data_for_test=mnist.test;
    accuraccy_rate=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(l4,axis=1),tf.argmax(y_,1)),dtype=tf.float32))
    print('test accuracy : ',sess.run(accuraccy_rate,feed_dict={x:data_for_test.images,y_:data_for_test.labels,keep_rate:1.0}));




