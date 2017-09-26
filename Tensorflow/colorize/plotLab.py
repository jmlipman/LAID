import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.interactive(True)
import cv2,os
from skimage import color

# Number of discretized regions of LAB colorspace

mapp=np.zeros((220,220))
l = []
c=0
for name in os.listdir("forest2"):
        c+=1
        print(c,name)
        im = cv2.cvtColor(cv2.imread("forest2/"+name),cv2.COLOR_BGR2RGB)
        im2 = color.rgb2lab(im)
        for i in range(im2.shape[0]):
                for j in range(im2.shape[1]):
                        mapp[int(im2[i,j,1])+110,int(im2[i,j,2])+110]+=1

plt.figure()
plt.imshow(np.log(mapp))

np.save("distrMap_shit",mapp)


colors=np.zeros((220,220,3))
for i in range(-110,110):
        for j in range(-110,110):
                colors[i+110,j+110]=[50,i,j]
#mapp = cv2.cvtColor(mapp,cv2.COLOR_Lab2RGB)
#plt.imshow(color.lab2rgb(mapp))
plt.figure()
plt.imshow(color.lab2rgb(colors))
