import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from jmlipman.util import equally_spaced_indices
from jmlipman.tf import AnimateWeights
from jmlipman.ml import divideDatasets
import time

Range = [0,10]
Range_offset = 0.5
Total_samples = 30
Training_samples = 50
Outliers = 10

# Preparing the data
x = np.linspace(Range[0]+Range_offset,Range[1]-Range_offset,num=Training_samples)
y = np.array([x[j]+x[j] for j in range(Training_samples)])

outliers = np.zeros(Training_samples)
indices = equally_spaced_indices(Outliers,Training_samples)
outliers[indices]+=np.random.random()*10
y+=outliers+np.random.random(Training_samples)*2

data = np.stack((x,y),axis=1)
data = np.load("my-data.npy")
#np.save("my-data",data)

trainingData,testingData,_ = divideDatasets(data,[0.8,0.2,0])

# Graph constants
epochs = 100
# Batch size of one will make it noisier
b_size = 1

x = tf.placeholder("float",[None,1],name="input")
y = tf.placeholder("float",[None,1],name="output")
lambdaReg = 0.3
weightsNames = ["dense1/W:0","dense2/W:0","dense3/W:0","dense4/W:0"]

AW = AnimateWeights()

with tf.variable_scope("dense1") as scope:
	W = tf.get_variable("W",shape=[1,10],initializer=tf.contrib.layers.xavier_initializer(),regularizer=tf.contrib.layers.l2_regularizer(lambdaReg))
	b = tf.get_variable("b",initializer=tf.zeros(10))
	dense = tf.add(tf.matmul(x,W),b)

with tf.variable_scope("dense2") as scope:
	W = tf.get_variable("W",shape=[10,10],initializer=tf.contrib.layers.xavier_initializer(),regularizer=tf.contrib.layers.l2_regularizer(lambdaReg))
	b = tf.get_variable("b",initializer=tf.zeros(10))
	dense = tf.add(tf.matmul(dense,W),b)

with tf.variable_scope("dense3") as scope:
	W = tf.get_variable("W",shape=[10,1],initializer=tf.contrib.layers.xavier_initializer(),regularizer=tf.contrib.layers.l2_regularizer(lambdaReg))
	b = tf.get_variable("b",initializer=tf.zeros(1))
	dense = tf.add(tf.matmul(dense,W),b)

with tf.variable_scope("dense4") as scope:
	W = tf.get_variable("W",shape=[1,1],initializer=tf.contrib.layers.xavier_initializer(),regularizer=tf.contrib.layers.l2_regularizer(lambdaReg))
	b = tf.get_variable("b",initializer=tf.zeros(1))
	pred = tf.add(tf.matmul(dense,W),b,name="prediction")


reg_losses = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
cost = tf.reduce_sum(tf.abs(tf.subtract(pred, y)))+reg_losses
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
AW.record(weightsNames,sess=sess)

for e in range(epochs):
	for b in range(0,trainingData.shape[0],b_size):
		x_raw = trainingData[b:b+b_size,0:1]
		y_raw = trainingData[b:b+b_size,1:2]
		_, _, c = sess.run([optimizer,AW.iter(sess),cost], feed_dict={x: x_raw, y: y_raw})
	
#AW.save_weights("my-weights")

# Plot dimensions
AW.configure_plot([2,2])
# Total time
AW.configure_gif(5)


AW.generate_GIF("example.gif")

plt.figure()
totalError=0
#newdata = np.linspace(Range[0],Range[1],num=results.shape[0])
for i in range(testingData.shape[0]):
	x_raw = np.reshape(testingData[i,0],(1,1))
	y_raw = np.reshape(testingData[i,1],(1,1))
	mypred = sess.run(pred,feed_dict={x: x_raw})
	totalError+=abs(y_raw-mypred)

print(totalError)

'''
plt.scatter(data[:,0],data[:,1],marker="x",color="red")
plt.plot(newdata,results)
plt.axis([Range[0],Range[1],0,25])
plt.title("Samples and prediction (L1)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
'''
