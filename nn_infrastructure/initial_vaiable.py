import tensorflow  as tf;

#initialize params
def get_noisy_variable(shape):
    w=tf.truncated_normal(shape,stddev=0.1);
    return tf.Variable(w);

def get_constant_variable(shape):
    b=tf.constant(0.1,shape=shape);
    return tf.Variable(b);