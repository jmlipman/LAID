# debugging1.py - First example of debugging a Keras neural network
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
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import scipy.misc
import numpy as np
from scipy.signal import convolve2d as conv

# ReLU function
def relu_fun(x):
    return x*(x>0)

# Softmax function
def softmax_fun(z):
	z_exp = [np.exp(i) for i in z]
	sum_z_exp = sum(z_exp)
	return np.array([round(i/sum_z_exp, 3) for i in z_exp])

# Return whether two matrices, vectors or arrays contain the same values
# They must have the same dimensions
def same(x1,x2):
    return np.sum(x1==x2)==x1.size

# Performs maxpooling manually
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

# Note that the initializer chosen was "Ones". The reason is because it's easier to
# understand its outcome.
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape,
		 kernel_initializer=keras.initializers.Ones()))

model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=keras.initializers.Ones()))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer=keras.initializers.Ones()))
model.add(Dense(num_classes, activation='softmax', kernel_initializer=keras.initializers.Ones()))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
		metrics=['accuracy'])


## Part 0: Checking out the model
# We have 6 different layers (each of them were added by model.add(..))
print(len(model.layers)) # Output: 6

# We can have a look at the layers
print(model.layers)

# We can see a pretty summary of our model
print(model.summary())

# The first layer takes a batch of of 28x28 input.
# We know it's a batch because of the "None" which is determined at running time
# The last "1" dimension shows that we are using 2D images.
# If this number is different than 1, we would be using 3D images.
print(model.layers[0].input_shape) # Output: (None, 28, 28, 1)

# Output of the first layer, which corresponds to the input of the second layer
print(model.layers[0].output_shape) # Output: (None, 26, 26, 32)
print(model.layers[1].input_shape) # Output: (None, 26, 26, 32)

# Weights used in the first layer.
# 2 tensors: 1) 32 3x3 kernels that will convolve the input
#	     2) a vector of 32 elements corresponding to the bias
print(model.layers[0].weights)

# If we want to see the values of these two components we can simply
print(model.layers[0].get_weights())


## Part 1: Obtaining the output after the first layer+activation
# We want to retrieve the output of the first convolution layer after the activation function
# which corresponds to the first layer of our model.
fun = K.function([model.layers[0].input],[model.layers[0].output])
# A single input is presented
x_inp = np.reshape(x,(1,28,28,1))
layer_output = fun([x_inp])[0]

# Shape of the output
print(layer_output.shape) # Output: (1, 26, 26, 32)
# We take any of the 32 26x26 outputs because they are the same
layer_output = np.reshape(layer_output[0,:,:,0],(26,26))

# Now we manually do the same to check whether we obtain the same results
tmp1=conv(x,np.ones((3,3)),mode="valid") # 2D convolution
sol1=relu_fun(tmp1) # Relu activation function
print("Checking if Part1 is true: "+str(same(sol1,layer_output))) # Output: True
scipy.misc.imsave('out1.jpg', sol1)


# Part 2: Obtaining the output after the second layer+activation
fun = K.function([model.layers[0].input],[model.layers[1].output])
x_inp = np.reshape(x,(1,28,28,1))
layer_output = fun([x_inp])[0]
layer_output = np.reshape(layer_output[0,:,:,0],(24,24))
tmp2=conv(sol1,np.ones((3,3)),mode="valid") # 2D convolution
sol2=relu_fun(tmp2) # Relu activation function
# We have to multiply sol2 by 32 because each of the 32 resulting filters of the first
# layer are summed, and all contain the same values, and bias is zero.
sol2*=32

print("Checking if Part2 is true: "+str(same(sol2,layer_output))) # Output: True
scipy.misc.imsave('out2.jpg', sol2)

# Part 3: After third layer, the maxpooling
fun = K.function([model.layers[0].input],[model.layers[2].output])
x_inp = np.reshape(x,(1,28,28,1))
layer_output = fun([x_inp])[0]
layer_output = np.reshape(layer_output[0,:,:,0],(12,12))

sol3 = maxpooling(sol2,(2,2))
print("Checking if Part3 is true: "+str(same(sol3,layer_output))) # Output: True
scipy.misc.imsave('out3.jpg', sol3)

# Part 4: After flatten
fun = K.function([model.layers[0].input],[model.layers[3].output])
x_inp = np.reshape(x,(1,28,28,1))
layer_output = fun([x_inp])[0]

# Copy the same 64 times (64 same filters) and flatten it
sol4 = np.zeros((12,12,64))
for i in range(sol4.shape[2]):
	sol4[:,:,i] = sol3

sol4 = np.reshape(sol4,(1,sol4.size)) # Shape of (1, 9216)
print("Checking if Part4 is true: "+str(same(sol4,layer_output))) # Output: True


# Part 5: After Dense 128
fun = K.function([model.layers[0].input],[model.layers[4].output])
x_inp = np.reshape(x,(1,28,28,1))
layer_output = fun([x_inp])[0]

# Each of the 9216 neurons multiplies each of the 128 neurons (which are ones because they
# are initialized like that, and summed.
sol5=relu_fun(np.sum(sol4)*np.ones((128)))
print("Checking if Part5 is true: "+str(same(sol5,layer_output))) # Output: True

# Part 6: Final layer plus softmax
fun = K.function([model.layers[0].input],[model.layers[5].output])
x_inp = np.reshape(x,(1,28,28,1))
layer_output = fun([x_inp])[0]

# Each of the 9216 neurons multiplies each of the 128 neurons (which are ones because they
# are initialized like that, and summed.
sol6=np.sum(sol5)*np.ones((num_classes))
# The values are really big, so we need to uniformly reduce them for the softmax function to work
sol6/=sol6[0]
sol6=softmax_fun(sol6)
sol6=np.reshape(sol6,(1,num_classes))
print("Checking if Part6 is true: "+str(same(sol6.astype("float32"),layer_output))) # Output: True
