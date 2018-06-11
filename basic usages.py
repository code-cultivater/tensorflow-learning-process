import tensorflow  as tf;

#build graph
a=tf.constant([[1.,2.]]);
b=tf.constant([[2.0],[3.0]]);
product=tf.matmul(a,b);

#run a session
with tf.Session() as sess:
    result=sess.run([product]);
    print(result);

#assign specific device

with tf.Session() as sess:
    with tf.device("/cpu:0"):
        m=tf.constant(1.0);
        n=tf.constant(2.0);
        l=tf.add(m,n);
    result=sess.run(l);
    print(result);


#variables needed initialed

a=tf.Variable(1.0);
b=tf.Variable(2.0);
c=tf.add(a,b);
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer());
    result=sess.run(c);
    print(result);


#fetch operation using passing more paras to function 'run'
d=tf.multiply(b,c);
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer());
    result=sess.run([d,c]);
    print(result);


# feed operation using tf.placeholder()

a=tf.placeholder(tf.float32);
b=tf.placeholder(tf.float32);
c=tf.divide(a,b);
with tf.Session() as sess:
    result=sess.run(c,feed_dict={a:[5.],b:[2.0]});
    print(result);


#
matrix=tf.constant([[1,2,3],[4,5,6],[7,8,9]]);
a=tf.Variable([1,2,3]);
sum=tf.add(matrix,a);

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer());
    result=sess.run([sum,a,matrix])
    print(result);


