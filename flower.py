import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import *


np.random.seed(1)

X, Y= load_planar_dataset()
plt.scatter(X[0,:],X[1,:],c=Y,s=40,cmap=plt.cm.Spectral);

shape_X=X.shape
shape_Y=Y.shape
m = X.shape[1]

clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T,Y.T);

# Plot the decision boundary for logistic regression
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")

# Print accuracy
LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")

def layer_sizes(X,Y):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    n_h = 4
    return (n_x,n_h,n_y)

def initialize_parameters(n_x,n_h,n_y):
    np.random.seed(2)
    W1 =np.random.randn(n_h,n_x)
    b1 =np.zeros((n_h,1))
    W2 =np.random.randn(n_y,n_h)
    b2 =np.zeros((n_y,1))
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def forward_propagation(X,parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1,X)+b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1)+b2
    A2 = sigmoid(Z2)
    assert (A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache
def compute_cost(A2, Y, parameters):
    m = Y.shape[1]
    logrobs = np.multiply(Y,np.log(A2))+np.multiply(np.log(1-A2),1-Y)
    cost=-np.sum(logrobs)/m
    cost = float(np.squeeze(cost))

    assert (isinstance(cost, float))

    return cost

def backward_propagation(parameters, cache, X, Y):
    m=X.shape[1]

    W1=parameters["W1"]
    W2=parameters["W2"]

    A1=cache["A1"]
    A2=cache["A2"]
    Z1=cache["Z1"]
    dZ2 = A2-Y
    dW2 = np.dot(dZ2,A1.T)/m
    db2 = np.sum(dZ2,axis=1,keepdims=True)/m
    dZ1 = np.dot(W2.T,dZ2)*(1-np.power(np.tanh(Z1),2))
    dW1 = np.dot(dZ1,X.T)/m
    db1 = np.sum(dZ1,axis=1,keepdims=True)/m
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads

def update_parameters(parameters,grads,learning_rate = 1.2):
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]

    dW1=grads["dW1"]
    dW2 = grads["dW2"]
    db1 = grads["db1"]
    db2 = grads["db2"]

    W1=W1-learning_rate*dW1
    W2 = W2 - learning_rate * dW2
    b1 = b1 - learning_rate * db1
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def nn_model(X,Y,n_h,num_iterations =1000,print_cost=False):
    np.random.seed(3)
    n_x=layer_sizes(X,Y)[0]
    n_y=layer_sizes(X,Y)[2]
    parameters=initialize_parameters(n_x,n_h,n_y)
    for i in range(0,num_iterations):
        A2,cache=forward_propagation(X,parameters)
        cost=compute_cost(A2,Y,parameters)
        grads=backward_propagation(parameters,cache,X,Y)
        parameters=update_parameters(parameters,grads,learning_rate=1.2)
        if print_cost and i%1000 ==0:
            print("Cost after iteration %i:%f" %(i,cost))
    return parameters

def predict(parameters,X):
    A2,cache=forward_propagation(X,parameters)
    predictions=(A2>0.5)*1
    return predictions

# Build a model with a n_h-dimensional hidden layer
parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)
# Print accuracy
predictions = predict(parameters, X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')
# Plot the decision boundary
# plt.figure(figsize=(16, 32))
# hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
# for i, n_h in enumerate(hidden_layer_sizes):
#     plt.subplot(5, 2, i+1)
#     plt.title('Hidden Layer of size %d' % n_h)
#     parameters = nn_model(X, Y, n_h, num_iterations = 5000)
#     plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
#     predictions = predict(parameters, X)
#     accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
#     print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))

# Datasets
# noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
#
# datasets = {"noisy_circles": noisy_circles,
#             "noisy_moons": noisy_moons,
#             "blobs": blobs,
#             "gaussian_quantiles": gaussian_quantiles}
#
# ### START CODE HERE ### (choose your dataset)
# dataset = "noisy_moons"
# ### END CODE HERE ###
#
# X, Y = datasets[dataset]
# X, Y = X.T, Y.reshape(1, Y.shape[0])
#
# # make blobs binary
# if dataset == "blobs":
#     Y = Y%2
#
# # Visualize the data
# plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()