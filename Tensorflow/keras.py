import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
#from tensorflow.keras.utils import layer_utils
#from tensorflow.keras.utils.data_utils import get_file
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
#from tensorflow.keras.utils.vis_utils import model_to_dot
from tensorflow.keras.utils import plot_model
from kt_utils import *
import tensorflow.compat.v1 as tf
import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

def HappyModel(input_shape):
    X_input=Input(input_shape)
    X=ZeroPadding2D((3,3))(X_input)

    X=Conv2D(32,(7,7),strides=(1,1),name='conv0')(X)
    X=BatchNormalization(axis=3,name='bn0')(X)
    X=Activation('relu')(X)

    X=MaxPooling2D((2,2),name='max_pool')(X)
    X=Flatten()(X)
    X=Dense(1,activation='sigmoid',name='fc')(X)

    model=Model(inputs=X_input,outputs=X,name='HappyModdel')
    return model



happyModel=HappyModel(X_train.shape[1:])
happyModel.compile(optimizer="Adam",loss="binary_crossentropy",mertics=["accuracy"])
happyModel.fit(x=X_train,y=Y_train,epochs=15,batch_size=16)
preds = happyModel.evaluate(x = X_test, y = Y_test)
print()
#print ("Loss = " + str(preds[0]))
#print ("Test Accuracy = " + str(preds[1]))
# img_path = 'images/Photo-1.jpeg'
# ### END CODE HERE ###
# img = image.load_img(img_path, target_size=(64, 64))
# imshow(img)
#
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
# print("adadd"+"\n")
# print(happyModel.predict(x))



