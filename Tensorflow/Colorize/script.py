import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.interactive(True)
import cv2,os
from scipy import stats
from skimage import color
import tensorflow as tf
import random

tf.reset_default_graph()

eps=10e-9
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

def softmax(z):
        return tf.exp(z)/tf.reduce_sum(tf.exp(z))
def gaussianFilter(kernlen=21, nsig=3):
        """Returns a 2D Gaussian kernel array."""

        interval = (2*nsig+1.)/(kernlen)
        x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
        kern1d = np.diff(stats.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw/kernel_raw.sum()
        return kernel

def lab2distr(Y):
        ''' This function will convert the NxNx2 distr. in a soft-encoding
            distribution NxNxQ


        '''
        # 2D Gaussian parameters
        kernel=5
        sigma=2
        # I could normalize the sigma so that the center has a value of 1.
        g = gaussianFilter(kernel,sigma)
        res = np.zeros((Y.shape[0],Y.shape[0],Q))
        for i in range(Y.shape[0]):
                for j in range(Y.shape[1]):
                        tmp = np.zeros((EqMapSmall.shape[0]+(kernel/2)*2,
                                        EqMapSmall.shape[1]+(kernel/2)*2))
                        # Get the a,b values that will be like coordinates
                        # and shift the values to fit the matrix.
                        a,b=Y[i,j,:]; a=int(a)+DistrMap.shape[0]/2;
                        b=int(b)+DistrMap.shape[0]/2;

                        colorClass = EqMap[a,b]
                        if colorClass<1:
                                print("This is fucked up!")

                        centerX,centerY = np.where(EqMapSmall==colorClass)
                        centerX = centerX[0]+kernel/2
                        centerY = centerY[0]+kernel/2

                        tmp[centerX-kernel/2:centerX+kernel/2+1,
                            centerY-kernel/2:centerY+kernel/2+1]=g
                        # Remove the borders I added before
                        tmp = tmp[kernel/2:-1*(kernel/2),kernel/2:-1*(kernel/2)]
                        #return tmp
                        # Now I flatten it and take the significant bins.
                        res[i,j,:]=tmp[C1,C2]
        return res

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

def mapWeights(batch):
        res = np.zeros((batch.shape[0],batch.shape[1],batch.shape[2]))
        for i in range(batch.shape[0]):
                for j in range(batch.shape[1]):
                        for k in range(batch.shape[2]):
                                ind1=(int(batch[i,j,k,0])+110)/WinMap
                                ind2=(int(batch[i,j,k,1])+110)/WinMap
                                res[i,j,k] = FinalWeights[ind1,ind2]

        return res


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
        #cost = tf.reduce_mean(ZW*-tf.reduce_sum(Y*tf.log(pred),3)) #16,16,16
        # I get nans
        # When I am getting nans it is usually either of the three:
        # - batch size too small (in your case then just 1)
        # - log(0) somewhere
        #- learning rate too high and uncapped gradients
        cost = tf.reduce_sum(ZW*-tf.reduce_sum(Y*tf.log(pred+eps),3)) #16,16,16

        train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

        return (X,Y,ZW),train_step,cost,pred

(X,Y,ZW),train_step,cost,pred=getModel()

saver = tf.train.Saver()

with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Training
        epochs = 10
        b_size = 32
        images = []
        for filename in os.listdir("dataset"):
                im = cv2.cvtColor(cv2.imread("dataset/"+filename),cv2.COLOR_BGR2RGB)
                images.append(color.rgb2lab(im))

        iters = [(x,y,z) for z in range(0,256,32) for y in range(0,256,32) for x in range(len(images))]

        for ep in range(epochs):
                random.shuffle(iters)
                cc=0
                print("Epoch: {0}.".format(ep))
                for batch in range(0,len(images),b_size):
                        cc+=1
                        input_batch = np.zeros((b_size,32,32,1))
                        labels_batch = np.zeros((b_size,16,16,Q))
                        colors_batch = np.zeros((b_size,16,16,2))
                        for index in range(b_size):
                                imID,i,j = iters[index]
                                piece = images[imID][i:i+32,j:j+32,:]
                                input_batch[index,:,:,0] = piece[:,:,0]
                                labels_batch[index,:,:,:] = np.reshape(lab2distr(piece[8:-8,8:-8,1:]),(1,16,16,Q))
                                colors_batch[index,:,:,:] = np.reshape(piece[8:-8,8:-8,1:],(1,16,16,2))
                        #print(colors_batch)
                        [_,c]=sess.run([train_step,cost],feed_dict={X:input_batch,Y:labels_batch,ZW:mapWeights(colors_batch)})
                        print(c)


        saver.save(sess,"test-model-good")
        # Testing
        for kk in range(10):
                ind = int(random.uniform(0,len(images)))
                res = np.zeros((256,256,3))
                res[:,:,0] = images[ind][:,:,0]
                lala=[];lolo=[]
                for i in range(8,256-8,16):
                        for j in range(8,256-8,16):
                                inp = np.reshape(images[ind][i-8:i+24,j-8:j+24,0],(1,32,32,1))

                                [p]=sess.run([pred],feed_dict={X:inp})
                                p = np.reshape(p,(16,16,Q))
                                lala.append(p)
                                lolo.append(inp)
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
