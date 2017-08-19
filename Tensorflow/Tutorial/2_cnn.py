# Author: Juan Miguel Valverde Martinez
# Date: 19 August 2017
# Youtube tutorial link: https://www.youtube.com/watch?v=btfKkOB5djQ
# Index: http://laid.delanover.com/tensorflow-tutorial/

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)

x_raw,y_raw = mnist.train.next_batch(40)
# Constants and variables
n_classes = 10
img_width = 28
img_height = 28

# For ipython
tf.reset_default_graph()

W = { "h1": tf.Variable(tf.ones([3,3,1,32])),
        "h2": tf.Variable(tf.ones([3,3,32,64])),
        "h3": tf.Variable(tf.ones([12*12*64,128])),
        "out": tf.Variable(tf.ones([128,n_classes])),
}

b = { "b1": tf.Variable(tf.zeros([32])),
        "b2": tf.Variable(tf.zeros([64])),
        "b3": tf.Variable(tf.zeros([128])),
        "bout": tf.Variable(tf.zeros([n_classes])),
}

# Network
x = tf.placeholder("float", [None, img_width*img_height])
y = tf.placeholder("float", [None, n_classes])

xi = tf.reshape(x,[-1,28,28,1])

l1 = tf.nn.conv2d(xi,W["h1"],strides=[1,1,1,1],padding="VALID")
l1_b = tf.nn.bias_add(l1,b["b1"])
l1_act = tf.nn.relu(l1_b)

l2 = tf.nn.conv2d(l1_act,W["h2"],strides=[1,1,1,1],padding="VALID")
l2_b = tf.nn.bias_add(l2,b["b2"])
l2_act = tf.nn.relu(l2_b)

l3_max = tf.nn.max_pool(l2_act,[1,2,2,1],strides=[1,2,2,1],padding="VALID")

# Dense
l4 = tf.reshape(l3_max,[-1,12*12*64])
l4 = tf.matmul(l4,W["h3"])
l4_b = tf.nn.bias_add(l4,b["b3"])
l4_act = tf.nn.relu(l4_b)

l5 = tf.matmul(l4,W["out"])+b["bout"]
out_act = tf.nn.softmax(l5)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=l5,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        pred = l1_b.eval({x: x_raw})
        #print(pred)
        '''
        for i in range(100):
                print(i)
                kk = sess.run(optimizer,feed_dict={x: x_raw,y: y_raw})

        pred = out_act.eval({x: x_raw})
        #print(pred,y_raw)
        
        m = W["h1"].eval()
        '''
