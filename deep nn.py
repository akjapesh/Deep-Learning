import numpy as np
import h5py
import matplotlib.pyplot as plt
from dnn_app_utils_v3 import *
from testCases_v4a import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward
import time
import scipy
from PIL import Image
from scipy import ndimage

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)
train_x_orig,train_y,test_x_orig,test_y,classes=load_data()

m_train=train_x_orig.shape[0]
num_px=train_x_orig.shape[1]
m_test=test_x_orig.shape[0]

train_x_flatten=train_x_orig.reshape(m_train,-1).T
test_x_flatten=test_x_orig.reshape(m_test,-1).T
train_x=train_x_flatten/255
test_x=test_x_flatten/255




def initialize_parameters(n_x,n_h,n_y):
    np.random.seed(1)
    W1=np.random.randn(n_h,n_x)*0.01
    W2 = np.random.randn(n_y, n_h) * 0.01
    b1=np.zeros((n_h,1))
    b2=np.zeros((n_y,1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    L=len(layer_dims)
    parameters={}
    for l in range(1,L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1])  *0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))


    assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
    assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

def linear_forward(A,W,b):
    Z=np.dot(W,A)+b
    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache

def linear_activation_forward(A_prev,W,b,activation):
    if activation=="sigmoid":
        Z,linear_cache=linear_forward(A_prev,W,b)
        A,activation_cache=sigmoid(Z)
    elif activation=="relu":
        Z,linear_cache=linear_forward(A_prev,W,b)
        A,activation_cache=relu(Z)
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X,parameters):
    caches=[]
    A=X
    L=len(parameters)//2
    for l in range(1,L):
        A_prev=A

        A,cache=linear_activation_forward(A_prev,parameters["W"+str(l)],parameters["b"+str(l)],"relu")
        caches.append(cache)

    AL,cache=linear_activation_forward(A,parameters["W"+str(L)],parameters["b"+str(L)],"sigmoid")
    caches.append(cache)
    assert (AL.shape == (1, X.shape[1]))

    return AL, caches

def compute_cost(AL,Y):
    m=Y.shape[1]
    cost=-np.sum(np.multiply(Y,np.log(AL))+np.multiply(1-Y,np.log(1-AL)))/m
    cost=np.squeeze(cost)
    assert (cost.shape == ())

    return cost

def linear_backward(dz,cache):
    A_prev,W,b=cache
    m=A_prev.shape[1]
    dW=np.dot(dz,A_prev.T)/m
    db=np.sum(dz,axis=1,keepdims=True)/m
    dA_prev=np.dot(W.T,dz)
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db
# Set up some test inputs

def linear_activation_backward(dA,cache,activation):
    linear_cache, activation_cache = cache

    if activation =="relu":
        dZ=relu_backward(dA,activation_cache)
        dA_prev,dW,db=linear_backward(dZ,linear_cache)
    elif activation=="sigmoid":
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL,Y,caches):
    grads={}
    L=len(caches)
    m=AL.shape[1]
    Y=Y.reshape(AL.shape)

    dAL=-(np.divide(Y,AL)-np.divide(1-Y,1-AL))

    current_cache = caches[L-1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,current_cache,"sigmoid")
    for l in reversed(range(L - 1)):

        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)],current_cache,"relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp


    return grads
def update_parameters(parameters, grads, learning_rate):
    L=len(parameters)//2
    for l in range(L):
        parameters["W"+str(l+1)]=parameters["W"+str(l+1)]-learning_rate*grads["dW"+str(l+1)]
        parameters["b" + str(l + 1)]=parameters["b"+str(l+1)]-learning_rate*grads["db"+str(l+1)]

    return parameters

n_x=test_x.shape[0]
n_h=7
n_y=1
layers_dims=(n_x,n_h,n_y)

def two_layer_model(X,Y,layers_dims,learning_rate=0.0075,num_iterations=30,print_cost=False):
    np.random.seed(1)
    grads={}
    costs=[]
    m=X.shape[1]
    (n_x,n_h,n_y)=layers_dims
    parameters=initialize_parameters(n_x,n_h,n_y)

    W1=parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0,num_iterations):
        A1,cache1=linear_activation_forward(X,W1,b1,"relu")
        A2,cache2=linear_activation_forward(A1,W2,b2,"sigmoid")

        cost=compute_cost(A2,Y)

        dA2=-(np.divide(Y,A2)-np.divide(1-Y,1-A2))

        dA1,dW2,db2=linear_activation_backward(dA2,cache2,"sigmoid")
        dA0,dW1,db1=linear_activation_backward(dA1,cache1,"relu")

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        parameters=update_parameters(parameters,grads,learning_rate)

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)

        # plot the cost

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

layer_dims=[12288,20,7,5,1]
def L_layer_model(X,Y,layer_dims,learning_rate=0.0075,num_iterations=3000,print_cost=False):
    np.random.seed(1)
    costs=[]
    parameters=initialize_parameters_deep(layers_dims)
    for i in range(0,num_iterations):
        AL,caches=L_model_forward(X,parameters)

        cost=compute_cost(AL,Y)

        grads=L_model_backward(AL,Y,caches)
        parameters=update_parameters(parameters,grads,learning_rate)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

            # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


