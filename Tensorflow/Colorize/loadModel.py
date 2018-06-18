import tensorflow as tf
import os, cv2
from skimage import color
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.interactive(True)
import cv2,os
from scipy import stats
import random


# Distributed map showing where most of the values are found
DistrMap = np.load("distrMap_all.npy") # 220x220
# Width and height of the window used to divide the distrMap equally
WinMap = 5 # 220/10 -> 22. 22*22 = 484 tiles.
# Small map: contains the tiles already reduced
EqMapSmall=np.zeros((DistrMap.shape[0]/WinMap,DistrMap.shape[0]/WinMap))
# Contains the whole distributed map
EqMap=np.zeros_like(DistrMap)
# Probability distribution used to weight the cross-entropy during the opt.
P_tilde=np.zeros((DistrMap.shape[0]/WinMap,DistrMap.shape[0]/WinMap))
# Contains the whole distributed map

# Temperature: 0 -> one-hot vector
#              1 -> more mean/saturated
T=0.01


# Number of discretized regions of LAB colorspace
c=0
for i in range(0,DistrMap.shape[0],WinMap):
        for j in range(0,DistrMap.shape[0],WinMap):
                suma = np.sum(DistrMap[i:i+WinMap,j:j+WinMap])
                if suma>0:
                        c+=1
                        EqMap[i:i+WinMap,j:j+WinMap]=c
                        EqMapSmall[i/WinMap,j/WinMap]=c
                        P_tilde[i/WinMap,j/WinMap]=suma
# Number of tiles
Q = c

# Coordinates where the significant bins can be found
C1,C2=np.where(EqMapSmall!=0)
P_tilde = P_tilde[C1,C2]
Lambda = 0.5
Weights = 1/((1-Lambda)*P_tilde+(Lambda/Q))
# Normalized such that np.sum(Weights*P_tilde)==1
Weights = Weights/np.sum(P_tilde*Weights)

FinalWeights = np.zeros_like(EqMapSmall)
FinalWeights[C1,C2]=Weights

def distr2lab(Z):
        ''' This function converts from NxNxQ to NxNx3 (original image).
                
            Arguments:
                Z: NxNxQ distribution
                bw: black and white image (L component from Lab)

        '''
        image = np.zeros((Z.shape[0],Z.shape[1],2))
        for i in range(Z.shape[0]):
                for j in range(Z.shape[1]):
                        l = Z[i,j,:]
                        #res2=np.exp(np.log(l)/T)/(np.sum(np.exp(np.log(l)/T)))
                        res2=l
                        matrix = np.zeros_like(EqMapSmall)
                        # I put the distribution back to the original colorspace
                        matrix[C1,C2]=res2

                        c1,c2=np.where(matrix!=0)
                        probs = matrix[matrix!=0]
                        colorX = np.sum(c1*(probs*10))/np.sum(probs*10)
                        colorY = np.sum(c2*(probs*10))/np.sum(probs*10)

                        colorX = colorX*WinMap-DistrMap.shape[0]/2
                        colorY = colorY*WinMap-DistrMap.shape[0]/2

                        image[i,j] = [colorX,colorY]

        return image
def getModel():
        X = tf.placeholder("float",[None, 32,32,1])
        Y = tf.placeholder("float",[None,16,16,Q])
        ZW = tf.placeholder("float",[None,16,16])

        step = int(Q/8)

        with tf.variable_scope("conv1") as scope:
                W = tf.get_variable("W",shape=[3,3,1,step],
                        initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable("b",initializer=tf.zeros([step]))
                l = tf.nn.bias_add(
                        tf.nn.conv2d(X,W,strides=[1,1,1,1],padding="VALID"),b)
                l_act = tf.nn.relu(l)

        for i in range(2,8):
                with tf.variable_scope("conv"+str(i)) as scope:
                        W = tf.get_variable("W",shape=[3,3,step*(i-1),step*i],
                                initializer=tf.contrib.layers.xavier_initializer())
                        b = tf.get_variable("b",initializer=tf.zeros([step*i]))
                        l = tf.nn.bias_add(
                                tf.nn.conv2d(l_act,W,strides=[1,1,1,1],padding="VALID"),b)
                        l_act = tf.nn.relu(l)

        with tf.variable_scope("conv8") as scope:
                W = tf.get_variable("W",shape=[3,3,step*7,Q],
                        initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable("b",initializer=tf.zeros([Q]))
                output = tf.nn.bias_add(
                        tf.nn.conv2d(l,W,strides=[1,1,1,1],padding="VALID"),b)
                #l_act = tf.nn.softmax(l)


        pred = tf.nn.softmax(output)
        # I should add the weights
        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=Y))

        # I guess i should reduce it
        cost = tf.reduce_sum(ZW*-tf.reduce_sum(Y*tf.log(pred),3)) #16,16,16
        # TODO: change Z for W that come directly from the input
                                #cost+=FinalWeights[Z[i,j,k,0],Z[i,j,k,1]]*init_cost[i,j,k]

        train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

        return (X,Y,ZW),train_step,cost,pred

(X,Y,ZW),train_step,cost,pred=getModel()


saver = tf.train.import_meta_graph('test-model-test.meta')

sess = tf.InteractiveSession()

saver.restore(sess,tf.train.latest_checkpoint('./'))


#sess.run(tf.global_variables_initializer())

# Testing

for num in range(5):
        filename=random.choice(os.listdir("dataset"))
        im2 = cv2.cvtColor(cv2.imread("dataset/"+filename),cv2.COLOR_BGR2RGB)
        im2 = color.rgb2lab(im2)

        res = np.zeros((256,256,3))
        res[:,:,0] = im2[:,:,0]
        for i in range(8,256-8,16):
                for j in range(8,256-8,16):
                        inp = np.reshape(im2[i-8:i+24,j-8:j+24,0],(1,32,32,1))
                        [p]=sess.run([pred],feed_dict={X:inp})
                        p = np.reshape(p,(16,16,Q))
                        dp = distr2lab(p)

                        res[i:i+16,j:j+16,1:] = dp
                        #print(dp.shape)

        res_converted = color.lab2rgb(res)
        res2 = np.zeros_like(res)
        res2[:,:,1:] = res[:,:,1:]
        res2[:,:,0] = 100
        res2 = color.lab2rgb(res2)
        plt.figure()
        plt.imshow(res_converted)
