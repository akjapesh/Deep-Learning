import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
#from tensorflow.keras.utils import layer_utils
#from tensorflow.keras.utils.data_utils import get_file
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
#from tensorflow.keras.utils.vis_utils import model_to_dot
from tensorflow.keras.utils import plot_model
from resnets_utils import *
from tensorflow.keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession



config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

tf.disable_eager_execution()


import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

def identity_block(X,f,filters,stage,block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)


    return X

def convolutional_block(X,f,filters,stage,block,s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    X=Conv2D(filters=F1,kernel_size=(1,1),strides=(s,s),kernel_initializer=glorot_uniform(seed=0),name=conv_name_base+'2a')(X)
    X=BatchNormalization(axis=3,name=bn_name_base+'2a')(X)
    X=Activation('relu')(X)


    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(seed=0),
               name=conv_name_base + '2b')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0), name=conv_name_base + '2c')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X_shortcut=Conv2D(filters=F3,kernel_size=(1,1),strides=(s,s),padding='valid',kernel_initializer=glorot_uniform(seed=0),name=conv_name_base+'1')(X_shortcut)
    X_shortcut=BatchNormalization(axis=3,name=bn_name_base+'1')(X_shortcut)
    X=Add()([X_shortcut,X])
    X = Activation('relu')(X)
    return X

def ResNet50(input_shape=(64,64,3),classes=6):
    X_input=Input(input_shape)
    X=ZeroPadding2D((3,3))(X_input)
    X=Conv2D(64,kernel_size=(7,7),strides=(2,2),name='conv1',kernel_initializer=glorot_uniform(seed=0))(X)
    X=BatchNormalization(axis=3,name='bn_conv1')(X)
    X=Activation('relu')(X)
    X=MaxPooling2D((3,3),strides=(2,2))(X)

    X=convolutional_block(X,f=3,filters=[64,64,256],stage=2,block='a',s=1)
    X=identity_block(X,3,[64,64,256],stage=2,block='b')
    X=identity_block(X,3,[64,64,256],stage=2,block='c')

    X=convolutional_block(X,f=3,filters=[128,128,512],s=2,stage=3,block='a')
    X=identity_block(X,3,[128,128,512],stage=3,block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    X = convolutional_block(X, f=3, filters=[256, 256, 1024], s=2, stage=4, block='a')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    X = convolutional_block(X, f=3, filters=[512,512,2048], s=2, stage=5, block='a')
    X = identity_block(X, 3, [512,512,2048], stage=5, block='b')
    X = identity_block(X, 3, [512,512,2048], stage=5, block='c')

    X=AveragePooling2D((2,2),name='avg_ppol')(X)
    X=Flatten()(X)
    X=Dense(classes,activation='softmax',name='fc'+str(classes),kernel_initializer=glorot_uniform(seed=0))(X)

    model=Model(inputs=X_input,outputs=X,name='Resnet50')
    return model
model = ResNet50(input_shape = (64, 64, 3), classes = 6)
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
model.fit(X_train,Y_train,epochs=2,batch_size=32)
preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))



