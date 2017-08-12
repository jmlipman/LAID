# debugging2.py - Second example of debugging a Keras neural network
# 
# This code gives an overview of how to debug a keras network.
#
# You can do with this code whatever you want. The main purpose is help
# people learning about this. Also, there is no warranty of any kind.
#
# Juan Miguel Valverde Martinez
# http://laid.delanover.com

from __future__ import print_function
import keras, math
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras import backend as K
import scipy.misc
import numpy as np
from scipy.signal import convolve2d as conv

# ReLU function
def relu_fun(x):
    return x*(x>0)

# Sigmoid function
def sigmoid_fun(x):
	return 1/(1+np.exp(-x))

# Softmax function
def softmax_fun(z):
	z_exp = [np.exp(i) for i in z]
	sum_z_exp = sum(z_exp)
	return np.array([round(i/sum_z_exp, 3) for i in z_exp])

# Return whether two matrices, vectors or arrays contain the same values
# They must have the same dimensions
def same(x1,x2):
    return np.sum(x1==x2)==x1.size

def maxpooling(x,stride):
	sol = np.zeros((x.shape[0]/stride[0],x.shape[1]/stride[1]))
	for i in range(0,x.shape[0],stride[0]):
		for j in range(0,x.shape[1],stride[1]):
			sol[i/stride[0],j/stride[1]] = np.max(x[i:i+stride[0],j:j+stride[1]])

	return sol
	
num_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Used sample. We make it binary
x = 1.0*(x_train[0,:,:,0]!=0)
scipy.misc.imsave('input.jpg', x)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 input_shape=input_shape,
		 kernel_initializer=keras.initializers.Ones()))
model.add(Activation("sigmoid"))

model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=keras.initializers.Ones()))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, kernel_initializer=keras.initializers.Ones()))
model.add(Activation("sigmoid"))
model.add(Dense(num_classes, kernel_initializer=keras.initializers.Ones()))
model.add(Activation("softmax"))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
		metrics=['accuracy'])


# The layers of the model were changed such that the first activation is not embedded into the first layer
# but it comprises an independent layer itself. 

print(model.summary())

# With and without the activation
fun_without = K.function([model.layers[0].input],[model.layers[0].output])
fun_with = K.function([model.layers[0].input],[model.layers[1].output])
# Input
x_inp = np.reshape(x,(1,28,28,1))
# Output
layer_output_without = fun_without([x_inp])[0]
layer_output_with = fun_with([x_inp])[0]

# Shape of the output
print(layer_output.shape) # Output: (1, 26, 26, 32)
# We take any of the 32 26x26 outputs because they are the same
layer_output_without = np.reshape(layer_output_without[0,:,:,0],(26,26))
layer_output_with = np.reshape(layer_output_with[0,:,:,0],(26,26))

# Now we manually do the same to check whether we obtain the same results
# Note: this implies that bias=0
sol1_without=conv(x,np.ones((3,3)),mode="valid") # 2D convolution
sol1_with=sigmoid_fun(sol1_without) # Relu activation function

# I need to round it because of the "approximation error"
print("Checking if without activation is true: "+str(same(np.round(sol1_without.astype("float32"),3),np.round(layer_output_without,3)))) # Output: True
print("Checking if with activation is true: "+str(same(np.round(sol1_with.astype("float32"),3),np.round(layer_output_with,3)))) # Output: True


# Part 2: Alternatively, using the network weights and bias.
fun_without = K.function([model.layers[0].input],[model.layers[0].output])
fun_with = K.function([model.layers[0].input],[model.layers[1].output])
x_inp = np.reshape(x,(1,28,28,1))
layer_output_without = fun_without([x_inp])[0]
layer_output_with = fun_with([x_inp])[0]


my_output_without = np.zeros((26,26,32))
# Weights and bias
W,b = model.layers[0].get_weights()
for i in range(32):
	my_output_without[:,:,i] = conv(x,W[:,:,0,i],mode="valid")+b[i]
my_output_without = np.reshape(my_output_without,(1,26,26,32))

print("Part 2: Checking if we got the same: "+str(same(my_output_without,layer_output_without)))



# Part 3:  Get output from the 7th layer (the second Dense) before the activation
fun = K.function([model.layers[0].input],[model.layers[6].output])
fun_sol = K.function([model.layers[0].input],[model.layers[7].output]) # solution
# Input
x_inp = np.reshape(x,(1,28,28,1))
layer_output = fun([x_inp])[0]
layer_output_sol = fun_sol([x_inp])[0]

W,b=model.layers[7].get_weights()
my_sol=np.dot(layer_output,W)+b

print("Part 3: Checking if we got the same: "+str(same(my_sol,layer_output_sol)))
