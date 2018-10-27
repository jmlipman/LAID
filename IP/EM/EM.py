# EM.py - Simple implementation of Expectation-Maximization algorithm
#
# You can do with this code whatever you want. The main purpose is help
# people learning about this. Also, there is no warranty of any kind.
#
# Juan Miguel Valverde Martinez
# http://laid.delanover.com

import numpy as np
import matplotlib.pyplot as plt
import random

# 2D Gaussian
def gauss2d(vector, cov, mean):
    denom = np.sqrt(np.power(2*np.pi, 2)*cov[0,0]*cov[1,1])
    sub = np.array([vector-mean])
    nume = np.exp((-1.0/2)*np.dot(np.dot(sub, np.linalg.inv(cov)),sub.T))
    return nume/denom


# Generate the data
# Number of data points per class, mean and variance
dataC1 = 400
dataC2 = 1000

mean1 = np.array([1, 2])
mean2 = np.array([5, 3])
cov_mat1 = np.array([[0.3, 0],[0, 0.5]])
cov_mat2 = np.array([[1, 0],[0, 1]])

# 3 cols corresponding to:
#  1: x
#  2: y
#  3: probability for this data point to belong to class 1
#  For class 2 it's simply 1-prev_prob
data = np.zeros((dataC1+dataC2, 2))
data[:dataC1, 0] = [random.gauss(mean1[0], cov_mat1[0,0]) for i in range(dataC1)]
data[:dataC1, 1] = [random.gauss(mean1[1], cov_mat1[1,1]) for i in range(dataC1)]
data[dataC1:, 0] = [random.gauss(mean2[0], cov_mat2[0,0]) for i in range(dataC2)]
data[dataC1:, 1] = [random.gauss(mean2[1], cov_mat2[1,1]) for i in range(dataC2)]
# In the beggining they all have the same probability to belong to any
# class.

titles = ["Initial Configuration", "Iteration 0", "Iteration 2", "Iteration 4",
          "Iteration 6", "Iteration 8"]

ax = plt.subplot(2,3,1)
ax.set_title(titles[0])
ax.scatter(data[:dataC1,0], data[:dataC1,1], color="b")
ax.scatter(data[dataC1:,0], data[dataC1:,1], color="r")

# Initial guesses
# Vectors of length 2 representing the mean/cov in the first/second dimension
mean1 = np.array([random.uniform(np.min(data[:,0]), np.max(data[:,0])),
                  random.uniform(np.min(data[:,1]), np.max(data[:,1]))])
mean2 = np.array([random.uniform(np.min(data[:,0]), np.max(data[:,0])),
                  random.uniform(np.min(data[:,1]), np.max(data[:,1]))])

cov_mat1 = np.array([[random.uniform(np.min(data[:,0]/2), np.max(data[:,0]/2)), 0],
                     [0, random.uniform(np.min(data[:,1]/2), np.max(data[:,1]/2))]])
cov_mat2 = np.array([[random.uniform(np.min(data[:,0]/2), np.max(data[:,0]/2)), 0],
                     [0, random.uniform(np.min(data[:,1]/2), np.max(data[:,1]/2))]])


prob_x1 = np.zeros(data.shape[0])
prob_x2 = np.zeros(data.shape[0])
c=1

for it in range(10):
    ### Expectation step
    # Gaussian
    for i in range(data.shape[0]):
        # Given the elements in class 1/2, what's the probability that a sample
        # belongs to that class
        prob_x1[i] = gauss2d(data[i], cov_mat1, mean1)
        prob_x2[i] = gauss2d(data[i], cov_mat2, mean2)

    # Given a sample, what's the probability it belongs to class 1/2
    prob_1 = np.sum(prob_x1)/np.sum([prob_x1, prob_x2])
    #prob_1 = 0.5
    prob_2 = 1-prob_1

    # Probability that samples belong to class 1
    prob_1x = (prob_x1*prob_1)/(prob_x1*prob_1 + prob_x2*prob_2)
    prob_2x = 1-prob_1x

    if it%2 == 0:
        c += 1
        ax = plt.subplot(2,3,c)
        group1 = data[prob_1x > 0.5]
        group2 = data[prob_1x <= 0.5]
        ax.set_title(titles[c-1])
        ax.scatter(group1[:,0], group1[:,1], color="r")
        ax.scatter(group2[:,0], group2[:,1], color="b")

    ### Maximization step
    # Calculate the means and covariance given the new probabilities
    mean1 = np.dot(prob_1x, data)/np.sum(prob_1x)
    mean2 = np.dot(prob_2x, data)/np.sum(prob_2x)

    cov1 = np.dot(prob_1x, np.power(data-mean1,2))/np.sum(prob_1x)
    cov_mat1 = np.array([[cov1[0], 0], [0, cov1[1]]])

    cov2 = np.dot(prob_2x, np.power(data-mean2,2))/np.sum(prob_2x)
    cov_mat2 = np.array([[cov2[0], 0], [0, cov2[1]]])


plt.show()
