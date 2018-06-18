import tensorflow as tf

class Model:

	def save_weights(self,path):
		""" This function saves the weights of the model.
		"""
		saver = tf.train.Saver()
		saver.save(self.sess,path+"weights")

	def load_weights(self,path):
		""" This function restores the weights into the model.
		"""
		saver = tf.train.import_meta_graph(path+"weights.meta")
		saver.restore(self.sess,tf.train.latest_checkpoint(path))
		#tf.summary.FileWriter("logs/",self.sess.graph)
		self.prediction = tf.get_default_graph().get_tensor_by_name("prediction:0")
		self.loss = tf.get_default_graph().get_tensor_by_name("loss/Mean:0")
		self.placeholder["input"] = tf.get_default_graph().get_tensor_by_name("input:0")
		self.placeholder["output"] = tf.get_default_graph().get_tensor_by_name("Placeholder:0")

	def create_loss(self):
		""" Set loss function.
		"""
		with tf.variable_scope("loss") as scope:
			y_true = self.placeholder["output"]
			y_pred = self.logits
		
			cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=y_true)
			self.loss = tf.reduce_mean(cross_entropy)

	def create_train_step(self):
		""" Create training part.
		"""
		with tf.variable_scope("optimization") as scope:
			grads = tf.gradients(self.loss,[self.placeholder["input"]])
			self.train_step = self.placeholder["input"] - tf.multiply(grads[0],1)
			

	def create_model(self):
		""" DL Model for the MSc thesis.
		"""
		iH,iW = self.conf["sizes"]
		init = self.conf["initializer"]

		self.placeholder["input"] = tf.Variable(tf.random_uniform((1,
				iH*iW)))
		self.placeholder["output"] = tf.placeholder(tf.float32,
				[1,self.conf["classes"]])

		x = tf.reshape(self.placeholder["input"],[-1, iH, iW, 1])
		
		with tf.variable_scope("conv1") as scope:
			# First 2D convolution, 
			W = tf.get_variable("W",shape=[3,3,1,32],initializer=init)
			b = tf.get_variable("b",initializer=tf.zeros([32]))

			conv = tf.nn.conv2d(x,W,strides=[1, 1, 1, 1],padding="VALID")
			pre_act = tf.nn.bias_add(conv,b)
			act = tf.nn.relu(pre_act)


		with tf.variable_scope("conv2") as scope:
			W = tf.get_variable("W",shape=[3,3,32,64],initializer=init)
			b = tf.get_variable("b",initializer=tf.zeros([64]))

			conv = tf.nn.conv2d(act,W,strides=[1, 1, 1, 1],padding="VALID")
			pre_act = tf.nn.bias_add(conv,b)
			act = tf.nn.relu(pre_act)

		# Maxpooling
		l3_mp = tf.nn.max_pool(act,[1,2,2,1],strides=[1,2,2,1],padding="VALID")

		# Dense
		l4 = tf.reshape(l3_mp,[-1, 12*12*64])

		with tf.variable_scope("dense1") as scope:
			W = tf.get_variable("W",shape=[12*12*64,128],initializer=init)
			b = tf.get_variable("b",initializer=tf.zeros([128]))

			dense = tf.matmul(l4,W)+b
			act = tf.nn.relu(dense)

		with tf.variable_scope("dense2") as scope:
			W = tf.get_variable("W",shape=[128,self.conf["classes"]],initializer=init)
			b = tf.get_variable("b",initializer=tf.zeros([self.conf["classes"]]))

			self.logits = tf.matmul(act,W)+b

		self.prediction = tf.nn.softmax(self.logits,name="prediction")
