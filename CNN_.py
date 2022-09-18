import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from extra_keras_datasets import emnist
from tensorflow.keras.models import load_model
from imaje_resije import *
import cv2 as cv

#********************************************************************************************************************************************************************8
def model_emnist():
    (x_train, y_train), (x_test, y_test) = emnist.load_data(type='balanced')
    #mnist=tf.keras.datasets.mnist #Handwritten dataset
    #(x_train,y_train),(x_test,y_test)=mnist.load_data()

    #Split to trainig data and testing data

    x_train = tf.keras.utils.normalize(x_train,axis=1)
    x_test = tf.keras.utils.normalize(x_test,axis=1)

    #1 input layer,2 hidden layers,1 output layer

    #Normalise RGB and Grayscale to 0 and 1

    model= tf.keras.models.Sequential() #Basic neural network
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    #New layer(flatten for 1 dimensional)

    model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))#rectify linear unit
    model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
    #   To connect the neural layers

    model.add(tf.keras.layers.Dense(units=50,activation=tf.nn.softmax))#Scales the probability
    #To find the probabilty of a specific handwritten element

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    #Optimizer-->Algorithm to change attributes of neural network. Adam uses a gradient
    #Loss--> Prediction and calculation of error 


    model.fit(x_train,y_train,epochs=3) 
    #To train the model(epochs-->reptition of model)
    model.save('digits.model')


    accuracy,loss=model.evaluate(x_test,y_test)
    print(accuracy)
    print(loss)
#***************************************************************************************************************************************************************************************
def model():
    mnist=tf.keras.datasets.mnist 
    (x_train,y_train),(x_test,y_test)=mnist.load_data()


    x_train = tf.keras.utils.normalize(x_train,axis=1)
    x_test = tf.keras.utils.normalize(x_test,axis=1)


    model= tf.keras.models.Sequential() 
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.softmax))
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit(x_train,y_train,epochs=10) 

    loss,accuracy=model.evaluate(x_test,y_test)
    print(accuracy)
    print(loss)
    model.save('digits2.model')
#*********************************************************************************************************************************************************************************
def model_digits2_model():
     model=load_model("digits2.model")# to avoid repeated training

     image_resize_model("3.png")
     img= cv.imread("3.png")[:,:,0]
     img=np.invert(np.array([img]))
     prediction=model.predict(img)
     print('The result is probably:',np.argmax(prediction))
     plt.imshow(img[0],cmap=plt.cm.binary)
     plt.show() 

    
#*********************************************************************************************************************************************************************************
