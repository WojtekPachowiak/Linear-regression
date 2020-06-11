import numpy as np
import pandas as pd

data = pd.read_excel("C:/Users/Administrator/Downloads/mlr03.xls")
data = data.values

################################################################################

def featureNormalize(data):    # feature normalization (z-score)
    mu = np.mean(data, axis=0)        # mean of columns
    sigma = np.std(data, axis=0)      # standard deviation of columns
    dataNorm = (data - mu) /sigma            # z-score
    return dataNorm


def addParameterX0(X):
    X0 = np.ones((X.shape[0],1))        # (number of examples x 1) array of ones
    X = np.hstack((X0,X))               # stack arrays horizontally
    return X


def computeCost(X, y, theta, m):   #compute cost
    J = (1/(2*m)) * sum((np.dot(X,theta)-y)**2)
    return J


def gradientDescent(X, y, theta, alpha, num_iters, m):
    J_history = np.zeros((num_iters, 1))
    for iter in range(len(J_history)):
        error = np.dot(X, theta) - y
        theta = theta - ((alpha/m) * np.dot(X.T, error))
        J_history[iter] = computeCost(X, y, theta,m)
    return J_history, theta


def predict(X, theta):
    prediction = np.dot(X, theta)
    return prediction

################################################################################

    #number of examples
alpha = 0.01        #learning rate
num_iters = 1000     #number of iterations


dataNorm = featureNormalize(data)
X = dataNorm[:,:-1]
y = dataNorm[:,-1][:, np.newaxis]
m = y.shape[0]

X = addParameterX0(X)
theta = np.zeros((X.shape[1], 1))   # create parameters vector

J = computeCost(X, y, theta, m)
J_history, theta = gradientDescent(X, y, theta, alpha, num_iters, m)
print(J_history)
print()
print(theta)
print()
prediction = predict(np.array([1, 20,20,60]), theta)
print(prediction)
