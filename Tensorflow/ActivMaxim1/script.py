import tensorflow as tf
import numpy as np
import jmlipman.ml
import time,os
from Model import Model as MNIST_Model
import scipy.misc
import itertools

class ActivationMaximization(MNIST_Model):
	def __init__(self,config,weights=None):
		self.conf = config
		self.placeholder = {}
		self.sess = tf.Session()

		self.load_weights(weights)
		self.create_train_step()

	def train(self,target):
		""" This function trains the network.
		"""
		print("Training")
		ep = self.conf["epochs"]

		# Preprocess the data
		#np.random.seed(seed=42)
		#res = [np.random.random((28*28))]
		res = [np.zeros(28*28)]
		Y = jmlipman.ml.hotvector(np.array([target]),self.conf["classes"])

		np.save("res/other",res)
		for e in range(ep):
			[res,loss,pred] = self.sess.run([self.train_step,self.loss,self.prediction],
							feed_dict={self.placeholder["input"]: np.expand_dims(res[0],0),
												 self.placeholder["output"]: Y})
			print(e,loss,np.argmax(pred))
		res2 = np.reshape(res[0],(28,28))
		np.save("res/other",res2)
		scipy.misc.imsave("res/other.png",res2)


	def predict(self,data):
		pred = self.sess.run(self.prediction,feed_dict={self.placeholder["input"]: data})
		return pred

# Get the data and split it
#data = jmlipman.ml.getDataset("mnist")
#data = jmlipman.ml.splitDataset(data,[0.8,0.2,0])

# Configuration part
learning_rate = 0.1
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
initializer = tf.keras.initializers.he_uniform()
classes = 10
sizes = [28,28]
epochs = 10

config = {"optimizer": optimizer, "learning_rate": learning_rate,
					"classes": classes, "sizes": sizes, "epochs": epochs,
					"initializer": initializer }

AM = ActivationMaximization(config,weights="weights/")
AM.train(2)
