import tensorflow as tf

msg = tf.constant('Hello, TensorFlow')
print(msg.numpy())

a = tf.constant(1)
b = tf.constant(2)
print((a + b).numpy())
