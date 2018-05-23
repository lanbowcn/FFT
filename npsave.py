import numpy as np
import tensorflow as tf
sess=tf.Session()
A = np.array([[0.1,0.5,0.4,0.8]])
B = np.array([[0,1,0,1]])
D=(A*2>=1)
with tf.Session() as sess:
    C=sess.run(tf.equal(D, B))
    print(sess.run(tf.cast(C, tf.float32)))