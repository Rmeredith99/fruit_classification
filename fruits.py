import numpy as np
from PIL import Image
import skimage.transform
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import TensorBoard
from time import time
import os
import sys
from clusterone import get_data_path, get_logs_path


def preprocess(image):
	"""
	[preprocess] takes in an image array of size (n x m x 3) and
		returns an image of size (64 x 64 x 3) with a mean of 0 and
		standard deviation of 1
	[image] - a numpy array of size (n x m x 3) that represents an image
	"""
	img = skimage.transform.resize(image, (64, 64, 3))
	img = img / 255.
	output = np.zeros((64, 64, 3), dtype = float)
	for rgb in range(3):
		color = img[:,:,rgb]
		color = color - np.mean(color)
		color = color / np.std(color)
		output[:,:,rgb] = color
	return output


def index_to_one_hot(index):
	"""
	[index_to_one_hot] creates a one-hot array denoting the output
		of an image with index given by [index]
	[index] - a number in range 0 to 63 inclusive
	"""
	vector = np.zeros(64, dtype = float)
	vector[index] = 1.
	return vector


def get_data():
	"""
	[get_data] returns pre-processed data from the image dataset
		in the form of inputs and outputs for training, validation,
		and testing.
		Returns: x_train, y_train, x_val, y_val, x_test, y_test
	"""
	
	# Specifying data paths for the Clusterone platform
	# training_data_path = get_data_path(
	# 		dataset_name = 'rmeredith99/fruit_data',  # on ClusterOne
	# 		local_root = 'C:\Users\Ryan Meredith\Documents\github',  # path to local dataset
	# 		local_repo = 'fruit_data',  # local data folder name
	# 		path = 'Training'  # folder within the data folder
	# 		)
	# 
	# validation_data_path = get_data_path(
	# 		dataset_name = 'rmeredith99/fruit_data',  # on ClusterOne
	# 		local_root = 'C:\Users\Ryan Meredith\Documents\github',  # path to local dataset
	# 		local_repo = 'fruit_data',  # local data folder name
	# 		path = 'Validation'  # folder within the data folder
	# 		)

	# Variables to keep track of progress
	z = 0. # progess counter
	total_images = 31688.+10657.
	
	# Training data
	index = 0
	n = 0
	x_train = np.zeros((31688, 64, 64, 3), dtype = float)
	y_train = np.zeros((31688, 64), dtype = float)
	
	# Iterating through folders to retrieve and process images
	for folder in os.listdir("Training"):
		for f in os.listdir("Training" + "/" + folder):
			img =  np.asarray(Image.open("Training" + "/" + folder + "/" + f))
			img = preprocess(img)
			x_train[n,:,:,:] = img
			y_train[n,:] = index_to_one_hot(index)
			n += 1
			
			# reporting on progress
			progress = (z+1) / total_images * 100
			sys.stdout.write("Data progress: %d%%   \r" % (progress))
			sys.stdout.flush()
			z += 1
			
		index += 1
		
	# Validation data
	index = 0
	n = 0
	x_val = np.zeros((10657, 64, 64, 3), dtype = float)
	y_val = np.zeros((10657, 64), dtype = float)
	
	# Iterating through folders to retrieve and process images
	for folder in os.listdir("Validation"):
		for f in os.listdir("Validation" + "/" + folder):
			img =  np.asarray(Image.open("Validation" + "/" + folder + "/" + f))
			img = preprocess(img)
			x_val[n,:,:,:] = img
			y_val[n,:] = index_to_one_hot(index)
			n += 1
			
			# reporting on progress
			progress = (z+1) / total_images * 100
			sys.stdout.write("Data progress: %d%%   \r" % (progress))
			sys.stdout.flush()
			z += 1
			
		index += 1
	
	# Randomizes the data in the same way for x and y data
	np.random.seed(516)
	np.random.shuffle(x_train)
	np.random.seed(516)
	np.random.shuffle(y_train)
	
	np.random.seed(516)
	np.random.shuffle(x_val)
	np.random.seed(516)
	np.random.shuffle(y_val)
	
	# Taking some of the validation data for testing
	x_test = x_val[:1000]
	y_test = y_val[:1000]
	x_val = x_val[1000:]
	y_val = y_val[1000:]
	
	return x_train, y_train, x_val, y_val, x_test, y_test


if __name__ == "__main__":
	# Turn off Keras warnings
	os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
	
	# Collecting data
	print("Begin collecting data\n")
	x_train, y_train, x_val, y_val, x_test, y_test = get_data()
	print("Finished collecting data")
	
	# Definition of the convolutional neural network
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size = (2, 2)))
	
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size = (2, 2)))
	
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size = (2, 2)))
	
	model.add(Flatten())
	model.add(Dense(128, activation = 'relu'))
	model.add(Dropout(0.4))
	
	model.add(Dense(64, activation = 'softmax'))
	
	model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	
	# Setting up logs
	log_path = get_logs_path(r"C:\Users\Ryan Meredith\Documents\github\fruit_classification\logs\\")
	tensorboard = TensorBoard(log_dir = log_path.format(time()))
	
	# Running the model
	model.fit(x_train, y_train, epochs = 10, batch_size = 64, validation_data = (x_val, y_val), callbacks = [tensorboard])
	model.save_weights("test2.h5")
	
	# Evaluating test data
	score = model.evaluate(x_test, y_test)
	print('Test Loss:', score[0])
	print('Test Accuracy:', score[1])
	