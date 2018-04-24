# Author: Juan Miguel Valverde Martinez
# Date: 24 April 2018
# Youtube tutorial link: https://www.youtube.com/watch?v=XXXX
# Blog: http://laid.delanover.com/

import numpy as np
import tensorflow as tf
from tensorflow.python.keras._impl.keras.applications.vgg16 import VGG16
from tensorflow.python.keras._impl.keras import backend as K
from PIL import Image
from scipy.optimize import fmin_l_bfgs_b as opt
from scipy.misc import imsave

class StyleTransfer:

	def __init__(self,configuration,content_image,style_image):

		self.conf = configuration
		self.style_im = tf.Variable(self._loadImage(style_image))
		self.content_im = tf.Variable(self._loadImage(content_image))

		self.createModel()
		self.createLoss()

		# Important line. For using VGG16+imagenet weights we need to load
		# the session. If we don't and we do global_variables_initializer
		# the weights will be randomly initialized.
		self.sess = K.get_session()

	def createModel(self):
		# Random noise image
		self.combination_im = tf.Variable(tf.random_uniform((1,
								self.conf["height"],self.conf["width"],3)))

		input_tensor = tf.concat([self.content_im,self.style_im,
								self.combination_im],0)
		self.model = VGG16(input_tensor=input_tensor,
				include_top=False, weights="imagenet")


	def createLoss(self):
		layers = dict([(layer.name,layer.output) for layer in self.model.layers])
		
		# VGG16 Layers that will represent the content and style
		layer_content = layers[self.conf["layer_content"]]
		layers_style = [layers[i] for i in self.conf["layers_style"]]
		
		self.loss = tf.Variable(0.)

		# Content Loss
		self.loss = tf.add(self.loss,
			self.conf["weight_content"] * self.contentLoss(
				layer_content[0,:,:,:],layer_content[2,:,:,:]))
		
		# Style Loss
		for i in range(len(self.conf["layers_style"])):
			self.loss = tf.add(self.loss,
						  self.conf["weights_style"][i] * self.styleLoss(
							layers_style[i][1,:,:,:],layers_style[i][2,:,:,:]))

		
	def contentLoss(self,content,combination):
	    """ Loss function for the content (details of the picture)
    	"""
	    return tf.reduce_sum(tf.square(content-combination))


	def styleLoss(self,style,combination):
		""" Loss function for the style.
		"""
		h,w,d = style.get_shape()
		M = h.value*w.value
		N = d.value
		S=self._GramMatrix(style)
		C=self._GramMatrix(combination)
		return tf.reduce_sum(tf.square(S - C)) / (4. * (N ** 2) * (M ** 2))

	
	def train(self,lr,epochs):
		""" This function will minimize the difference between the randomly
			generated image and the content-style images.
			You can uncomment to use the optimizer you want. Currently
			ScipyOptimizer (L-BFGS-B algorithm) and GradietnDescent can be
			used.

			Learning rate is only used in gradient descent, but epochs are
			used in both. ScipyOptimizer needs considerably less epochs becuase
			it already iterates inside.
		"""

		# ScipyOptimizer
		# Starts here
		train_step = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
					var_list=[self.combination_im],options={'maxfun':20})

		for i in range(epochs):
			curr_loss = self.sess.run(self.loss)
			print("Iteration {0}, Loss: {1}".format(i,curr_loss))
			train_step.minimize(session=self.sess)
		self.finalOutput = self.sess.run(self.combination_im)
		# End test
		
			
		"""
		# GradientDescent
		# Starts here
		myopt = tf.train.GradientDescentOptimizer(learning_rate=lr)
		tmp_grad = myopt.compute_gradients(self.loss,var_list=[self.combination_im])
		train_step = myopt.apply_gradients(tmp_grad)
		#self.sess.run(tf.global_variables_initializer())

		for i in range(epochs):
			curr_loss = self.sess.run(self.loss)
			print("Iteration {0}, Loss: {1}".format(i,curr_loss))
			self.sess.run(train_step)
		self.finalOutput = self.sess.run(self.combination_im)
		# End my version
		"""

	def saveOutput(self,name):
		""" This function will simply save the obtained image.
			Use it after train().
		"""
		out = self.finalOutput.reshape((self.conf["height"],self.conf["width"],3))
		out = np.clip(out,0,255).astype('uint8')
		imsave(name,out)


	def _GramMatrix(self,x):
		""" Returns a Gram matrix.
		"""
		# The first axis corresponds to the number of filters
		features=tf.keras.backend.batch_flatten(tf.transpose(x,perm=[2,0,1]))
		gram=tf.matmul(features, tf.transpose(features))
		return gram


	def _loadImage(self,image):
		""" This function will load an image.
		"""
		loaded_image = Image.open(image)
		loaded_image = loaded_image.resize((self.conf["height"],self.conf["width"]))
		loaded_array = np.asarray(loaded_image,dtype="float32")
		loaded_array = np.reshape(loaded_array,(1,self.conf["height"],self.conf["width"],3))

		return loaded_array


	
# An advice in adjusting the settings:
# First play with weights_style setting weight_content to 0, and
# when you got a pattern/style you like, increase weight_content.
config = {"height": 512, "width": 512, "layer_content": "block2_conv2",
		"layers_style": ["block1_conv2","block2_conv2","block3_conv3",
						"block4_conv3","block5_conv3"],
		"weight_content": 0.01, "weights_style": [100,100,100,100,100]}


st = StyleTransfer(config,"utebo.jpg","gustav.jpg")
st.train(-1,20)
st.saveOutput("result.png")
