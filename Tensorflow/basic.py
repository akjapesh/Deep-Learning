import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

np.random.seed(1)
tf.disable_eager_execution()

# GRADED FUNCTION: linear_function

def linear_function():
    """
    Implements a linear function:
            Initializes X to be a random tensor of shape (3,1)
            Initializes W to be a random tensor of shape (4,3)
            Initializes b to be a random tensor of shape (4,1)
    Returns:
    result -- runs the session for Y = WX + b
    """

    np.random.seed(1)

    """
    Note, to ensure that the "random" numbers generated match the expected results,
    please create the variables in the order given in the starting code below.
    (Do not re-arrange the order).
    """
    ### START CODE HERE ### (4 lines of code)

    ### END CODE HERE ###

    # Create the session using tf.Session() and run it with sess.run(...) on the variable you want to calculate

    ### START CODE HERE ###
    with tf.Session() as sess:
        X = tf.constant(np.random.randn(3, 1), name="X")
        W = tf.constant(np.random.randn(4, 3), name="W")
        b = tf.constant(np.random.randn(4, 1), name="X")

        Y = tf.add(tf.matmul(W, X), b)
        result = sess.run(Y)
    ### END CODE HERE ###

    # close the session
        sess.close()

    return result


def sigmoid(z):
    with tf.Session() as sess:
        x=tf.placeholder(tf.float32,name="x")
        sigmoid = tf.sigmoid(x,name="sigmoid")
        result=sess.run(sigmoid,feed_dict={x:z})
    return  result

def cost(logists,label):
    with tf.Session() as sess:
        z=tf.placeholder(tf.float32,name="z")
        y=tf.placeholder(tf.float32,name="y")
        cost=tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=z)
        cost=sess.run(cost,feed_dict={y:label,z:logists})
        sess.close()
    return cost

def one_hot_matrix(labels,C):
    with tf.Session() as ses:
        C=tf.constant(C,name="C")
        one_hot_matrix=tf.one_hot(labels,C)
        one_hot=ses.run(one_hot_matrix)
        ses.close()
    return one_hot

def ones(shape):
    with tf.Session() as ses:
        ones=tf.ones(shape)
        ones=ses.run(ones)
        ses.close()
    return  ones

X_train_orig,Y_train_orig,X_test_orig,Y_test_orig,classes=load_dataset()
X_train_flatten=X_train_orig.reshape(X_train_orig.shape[0],-1).T
X_test_flatten=X_test_orig.reshape(X_test_orig.shape[0],-1).T
X_train=X_train_flatten/255
X_test=X_test_flatten/255
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)
def create_placeholders(n_x,n_y):
    X=tf.placeholder(tf.float32,shape=[n_x,None],name="X")
    Y=tf.placeholder(tf.float32,shape=[n_y,None],name="Y")
    return X,Y

def initialize_parameters():
    tf.set_random_seed(1)
    W1 = tf.get_variable("W1",[25,12288],initializer=tf.keras.initializers.glorot_normal(seed=1))
    b1 = tf.get_variable("b1",[25,1],initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2",[12,25],initializer=tf.keras.initializers.glorot_normal(seed=1))
    b2 = tf.get_variable("b2",[12,1],initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3",[6,12],initializer=tf.keras.initializers.glorot_normal(seed=1))
    b3 = tf.get_variable("b3",[6,1],initializer=tf.zeros_initializer())
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters

def forward_propagation(X,parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1,X),b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1),b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2),b3)

    return Z3



def compute_cost(Z3,Y):
    logits=tf.transpose(Z3)
    labels=tf.transpose(Y)
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
    return cost

def model(X_train,Y_train,X_test,Y_test,learning_rate=0.0001,num_epochs=1500,minibatch_size=32,print_cost=True):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)


        for epoch in range(num_epochs):
            epoch_cost=0
            num_minibatches =int(m/minibatch_size)
            seed=seed+1
            minibatches=random_mini_batches(X_train,Y_train,minibatch_size,seed)
            for minibatch in minibatches:
                (minibatch_x,minibatch_y)=minibatch
                _,minibatch_cost=sess.run([optimizer,cost],feed_dict={X:minibatch_x,Y:minibatch_y})
                epoch_cost=minibatch_cost/minibatch_size
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

                # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

            # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

            # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

            # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters
parameters = model(X_train, Y_train, X_test, Y_test)