import numpy as np
from PIL import Image
import skimage.transform
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras import backend
import os
import sys
from clusterone import get_data_path

def preprocess(image):
	img = skimage.transform.resize(image,(64, 64, 3))
	img = img/255.
	output = np.zeros((64,64,3),dtype=float)
	for rgb in range(3):
		color = img[:,:,rgb]
		color = color - np.mean(color)
		color = color / np.std(color)
		output[:,:,rgb] = color
	return output

def index_to_one_hot(index):
	vector = np.zeros(64,dtype = float)
	vector[index] = 1.
	return vector

def get_data():
	
	training_data_path = get_data_path(
			dataset_name = 'rmeredith99/fruit_data',  # on ClusterOne
			local_root = 'C:\Users\Ryan Meredith\Documents\github',  # path to local dataset
			local_repo = 'fruit_data',  # local data folder name
			path = 'Training'  # folder within the data folder
			)
	
	validation_data_path = get_data_path(
			dataset_name = 'rmeredith99/fruit_data',  # on ClusterOne
			local_root = 'C:\Users\Ryan Meredith\Documents\github',  # path to local dataset
			local_repo = 'fruit_data',  # local data folder name
			path = 'Validation'  # folder within the data folder
			)

	# variables to keep track of progress
	z = 0.
	total_images = 31688.+10657.
	
	# training data
	index = 0
	n = 0
	x_train = np.zeros((31688,64,64,3),dtype = float)
	y_train = np.zeros((31688,64),dtype = float)
	for folder in os.listdir(training_data_path):
		for f in os.listdir(training_data_path + "/" + folder):
			img =  np.asarray(Image.open(training_data_path + "/" + folder + "/" + f))
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
		
	# validation data
	index = 0
	n = 0
	x_val = np.zeros((10657,64,64,3),dtype = float)
	y_val = np.zeros((10657,64),dtype = float)
	for folder in os.listdir(validation_data_path):
		for f in os.listdir(validation_data_path + "/" + folder):
			img =  np.asarray(Image.open(validation_data_path + "/" + folder + "/" + f))
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
	
	# randomizes the data in the same way for x and y data
	np.random.seed(516)
	np.random.shuffle(x_train)
	np.random.seed(516)
	np.random.shuffle(y_train)
	
	np.random.seed(516)
	np.random.shuffle(x_val)
	np.random.seed(516)
	np.random.shuffle(y_val)
	
	# taking some of the validation data for testing
	x_test = x_val[:1000]
	y_test = y_val[:1000]
	x_val = x_val[1000:]
	y_val = y_val[1000:]
	
	return x_train, y_train, x_val, y_val, x_test, y_test

if __name__ == "__main__":
	# turn off Keras warnings
	os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
	
	print("Begin collecting data\n")
	x_train, y_train, x_val, y_val, x_test, y_test = get_data()
	print("Finished collecting data")
	
	# definition of the convolutional neural network
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3),activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	model.add(Conv2D(32, (3, 3),activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	model.add(Conv2D(32, (3, 3),activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	model.add(Flatten())
	model.add(Dense(128,activation='relu'))
	model.add(Dropout(0.4))
	
	model.add(Dense(64,activation='softmax'))
	
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	
	#model.load_weights("test1.h5")
	
	# running the model
	model.fit(x_train,y_train,epochs=10,batch_size=64,validation_data=(x_val,y_val))
	model.save_weights("test2.h5")
	
	# evaluating test data
	score = model.evaluate(x_test,y_test)
	print('Test accuracy:', score[0])
	print('Test error:', score[1])
	