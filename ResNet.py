from tensorflow.keras.datasets import mnist
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import build_montages
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
from collections import OrderedDict
from load_data import *

#*********************************************************************************************************************************************************************************
		
def ResNet():
	#building resnet model
	from tensorflow.keras.layers  import BatchNormalization
	from keras.layers.convolutional import Conv2D
	from keras.layers.convolutional import AveragePooling2D
	from keras.layers.convolutional import MaxPooling2D
	from keras.layers.convolutional import ZeroPadding2D
	from keras.layers.core import Activation
	from keras.layers.core import Dense
	from keras.layers import Flatten
	from keras.layers import Input
	from keras.models import Model
	from keras.layers import add
	from keras.regularizers import l2
	from keras import backend as K
	#data augumentation

  
	class ResNet:                 #basic resnet architecture
		@staticmethod
		def residual_module(data, K, stride, chanDim, red=False,
		reg=0.0001, bnEps=2e-5, bnMom=0.9):
		# the shortcut branch of the ResNet module should be
		# initialize as the input (identity) data
			shortcut = data

		# the first block of the ResNet module are the 1x1 CONVs
			bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps,
				momentum=bnMom)(data)
			act1 = Activation("relu")(bn1)
			conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False,
				kernel_regularizer=l2(reg))(act1)

		# the second block of the ResNet module are the 3x3 CONVs
			bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps,
				momentum=bnMom)(conv1)
			act2 = Activation("relu")(bn2)
			conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride,
				padding="same", use_bias=False,
				kernel_regularizer=l2(reg))(act2)

		# the third block of the ResNet module is another set of 1x1
		# CONVs
			bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps,
				momentum=bnMom)(conv2)
			act3 = Activation("relu")(bn3)
			conv3 = Conv2D(K, (1, 1), use_bias=False,
				kernel_regularizer=l2(reg))(act3)

		# if we are to reduce the spatial size, apply a CONV layer to
		# the shortcut
			if red:
				shortcut = Conv2D(K, (1, 1), strides=stride,
					use_bias=False, kernel_regularizer=l2(reg))(act1)

		# add together the shortcut and the final CONV
			x = add([conv3, shortcut])

		# return the addition as the output of the ResNet module
			return x

		@staticmethod
		def build(width, height, depth, classes, stages, filters,
			reg=0.0001, bnEps=2e-5, bnMom=0.9, dataset="cifar"):
			# initialize the input shape to be "channels last" and the
			# channels dimension itself
			inputShape = (height, width, depth)
			chanDim = -1

			# if we are using "channels first", update the input shape
			# and channels dimension
			if K.image_data_format() == "channels_first":
				inputShape = (depth, height, width)
				chanDim = 1

			# set the input and apply BN
			inputs = Input(shape=inputShape)
			x = BatchNormalization(axis=chanDim, epsilon=bnEps,
				momentum=bnMom)(inputs)

		# check if we are utilizing the CIFAR dataset
			if dataset == "cifar":
				# apply a single CONV layer
				x = Conv2D(filters[0], (3, 3), use_bias=False,
					padding="same", kernel_regularizer=l2(reg))(x)
	
			# check to see if we are using the Tiny ImageNet dataset
			elif dataset == "tiny_imagenet":
				# apply CONV => BN => ACT => POOL to reduce spatial size
				x = Conv2D(filters[0], (5, 5), use_bias=False,
					padding="same", kernel_regularizer=l2(reg))(x)
				x = BatchNormalization(axis=chanDim, epsilon=bnEps,
					momentum=bnMom)(x)
				x = Activation("relu")(x)
				x = ZeroPadding2D((1, 1))(x)
				x = MaxPooling2D((3, 3), strides=(2, 2))(x)

		# loop over the number of stages
			for i in range(0, len(stages)):
				# initialize the stride, then apply a residual module
				# used to reduce the spatial size of the input volume
				stride = (1, 1) if i == 0 else (2, 2)
				x = ResNet.residual_module(x, filters[i + 1], stride,
					chanDim, red=True, bnEps=bnEps, bnMom=bnMom)

			# loop over the number of layers in the stage
				for j in range(0, stages[i] - 1):
					# apply a ResNet module
					x = ResNet.residual_module(x, filters[i + 1],
						(1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)

			# apply BN => ACT => POOL
			x = BatchNormalization(axis=chanDim, epsilon=bnEps,
				momentum=bnMom)(x)
			x = Activation("relu")(x)
			x = AveragePooling2D((8, 8))(x)

		# softmax classifier
			x = Flatten()(x)
			x = Dense(classes, kernel_regularizer=l2(reg))(x)
			x = Activation("softmax")(x)

		# create the model
			model = Model(inputs, x, name="resnet")

			# return the constructed network architecture
			return model

#*********************************************************************************************************************************************************************************

	EPOCHS = 10 # Train mode for 10 epochs
	INIT_LR = 1e-1 # Take initial learning rate as 0.01
	BS = 256 # Taking batch size of 256
	le = LabelBinarizer()
	print("[INFO] compiling model...")
	opt = SGD(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS) # Using SGD(Schotastic Gradient Descent) as optimizer

	model = ResNet.build(32, 32, 1, len(le.classes_), (3, 3, 3),
	(64, 64, 128, 256), reg=0.0005)

	model.compile(loss="categorical_crossentropy", optimizer=opt,
		metrics=["accuracy"])



	import numpy as np

	import os
	checkpoint_path = "model/"
	checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
	H = model.fit(
		aug.flow(trainX, trainY, batch_size=BS),
		validation_data=(testX, testY),
		steps_per_epoch=len(trainX) // BS,
		epochs=EPOCHS,
		class_weight=classWeight,
		verbose=1,
    	    callbacks=[cp_callback])
	model.save('model.h5')


#*********************************************************************************************************************************************************************************




                                                 
