# homography.m - Simple homography example
# 
# This code implements an homography algorithm based on SVD.
#
# You can do with this code whatever you want. The main purpose is help
# people learning about this. Also, there is no warranty of any kind.
#
# Juan Miguel Valverde Martinez
# http://laid.delanover.com

import numpy as np
from scipy import misc

p1 = [(450,68),(527,99),(603,130),(543,312),(483,494),(423,443),(363,393),(406,230)]
p2 = [(364,96),(443,96),(522,96),(522,253),(522,411),(443,411),(364,411),(364,253)]
p1 = [(448,67),(602,129),(362,392),(483,494)]
p2 = [(386,78),(552,78),(386,453),(552,453)]

pointsTotal = len(p1)

im = misc.imread("img.jpg")

for i in range(4):
        im[p1[i][0]-2:p1[i][0]+3,p1[i][1]-2:p1[i][1]+3,:] = [255,0,0]
        im[p2[i][0]-2:p2[i][0]+3,p2[i][1]-2:p2[i][1]+3,:] = [0,255,0]

misc.imsave("or.png",im)

A = np.zeros((8,9))
# Homography matrix
for i in range(4): # Using the corners
        A[i*2,:] = [ p1[i][1], p1[i][0], 1, 0, 0, 0, -p2[i][1]*p1[i][1], -p2[i][1]*p1[i][0], -p2[i][1] ]
        A[i*2+1,:] = [0, 0, 0, p1[i][1], p1[i][0], 1, -p2[i][0]*p1[i][1], -p2[i][0]*p1[i][0], -p2[i][0] ]


# Alternative (provide worse results)
#[V,W] = np.linalg.eig(np.dot(A.T,A))
#m = W[:,-1]

[U,S,V]=np.linalg.svd(A)
m = V[-1,:]
H = np.reshape(m,(3,3))
#H=H.T


# It should be close to zero
print("This value should be close to zero: "+str(np.sum(np.dot(A,m))))

H = np.reshape(m,(3,3))

# This part will will calculate the X and Y offsets
bunchX=[]; bunchY=[]

tt = np.array([[1],[1],[1]])
tmp = np.dot(H,tt)
bunchX.append(tmp[0]/tmp[2])
bunchY.append(tmp[1]/tmp[2])

tt = np.array([[im.shape[1]],[1],[1]])
tmp = np.dot(H,tt)
bunchX.append(tmp[0]/tmp[2])
bunchY.append(tmp[1]/tmp[2])

tt = np.array([[1],[im.shape[0]],[1]])
tmp = np.dot(H,tt)
bunchX.append(tmp[0]/tmp[2])
bunchY.append(tmp[1]/tmp[2])

tt = np.array([[im.shape[1]],[im.shape[0]],[1]])
tmp = np.dot(H,tt)
bunchX.append(tmp[0]/tmp[2])
bunchY.append(tmp[1]/tmp[2])

refX1 = int(np.min(bunchX))
refX2 = int(np.max(bunchX))
refY1 = int(np.min(bunchY))
refY2 = int(np.max(bunchY))

# Final image whose size is defined by the offsets previously calculated
final = np.zeros((int(refY2-refY1),int(refX2-refX1),3))

# Iterate over the imagine to transform every pixel
for i in range(im.shape[0]):
        for j in range(im.shape[1]):

                tt = np.array([[j],[i],[1]])
                tmp = np.dot(H,tt)
                x1=int(tmp[0]/tmp[2])-refX1
                y1=int(tmp[1]/tmp[2])-refY1

                if x1>0 and y1>0 and y1<refY2-refY1 and x1<refX2-refX1:
                        final[y1,x1,:]=im[i,j,:]

misc.imsave("_tmp_final.png",final)
# Simple Interpolation
# Interpolate empty pixels from the original image, ignoring pixels outside (extrapolating)
Hi = np.linalg.inv(H)
for i in range(final.shape[0]):
        for j in range(final.shape[1]):
                if sum(final[i,j,:])==0:
                        tt = np.array([[j+refX1],[i+refY1],[1]])
                        tmp = np.dot(Hi,tt)
                        x1=int(tmp[0]/tmp[2])
                        y1=int(tmp[1]/tmp[2])

                        if x1>0 and y1>0 and x1<im.shape[1] and y1<im.shape[0]:
                                final[i,j,:] = im[y1,x1,:]


misc.imsave("final.png",final)
