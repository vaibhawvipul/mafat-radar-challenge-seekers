#!/usr/bin/env python
# coding: utf-8

# #**MAFAT Radar Challenge - Baseline Model**

# AUC - 0.8078

import os
import random
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, GlobalAveragePooling2D, Concatenate, BatchNormalization
from tensorflow.keras import Input as inp 
from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from sklearn.metrics import roc_auc_score, roc_curve, auc
from matplotlib.colors import LinearSegmentedColormap
from termcolor import colored
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Set seed for reproducibility of results
seed_value = 0
os.environ['PYTHONHASHSEED']=str(seed_value)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

tf.compat.v1.reset_default_graph()

random.seed(seed_value)
np.random.seed(seed_value)
#tf.random.set_seed(seed_value)
tf.random.set_random_seed(seed_value)

# Configure a new global `tensorflow` session
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, log_device_placement=True)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = False)
tf.compat.v1.keras.backend.set_session(sess)

print(tf.__version__)

# Functions for loading the data
def load_data(file_path):
	"""
	Reads all data files (metadata and signal matrix data) as python dictionary,
	the pkl and csv files must have the same file name.

	Arguments:
		file_path -- {str} -- path to the iq_matrix file and metadata file

	Returns:
		Python dictionary
	"""
	pkl = load_pkl_data(file_path)
	meta = load_csv_metadata(file_path)
	data_dictionary = {**meta, **pkl}
	
	for key in data_dictionary.keys():
		data_dictionary[key] = np.array(data_dictionary[key])

	return data_dictionary


def load_pkl_data(file_path):
	"""
	Reads pickle file as a python dictionary (only Signal data).

	Arguments:
		file_path -- {str} -- path to pickle iq_matrix file

	Returns:
		Python dictionary
	"""
	path = os.path.join(file_path + '.pkl')
	with open(path, 'rb') as data:
		output = pickle.load(data)
	return output


def load_csv_metadata(file_path):
	
	"""
	Reads csv as pandas DataFrame (only Metadata).

	Arguments:
		file_path -- {str} -- path to csv metadata file

	Returns:
		Pandas DataFarme
	"""
	path = os.path.join(file_path + '.csv')
	with open(path, 'rb') as data:
		output = pd.read_csv(data)
	return output

# ## **Data Preprocessing**
# **The preprocessing main steps:**   
# 1. Applying [DFT](https://en.wikipedia.org/wiki/Discrete_Fourier_transform) (Discrete Fourier transform) by using the fast Fourier transform algorithm ([FFT](https://en.wikipedia.org/wiki/Fast_Fourier_transform)) and [Hann function](https://www.mathworks.com/help/signal/ref/hann.html) to smooth the I/Q matrix. Afterward, calculating the absolute value of the I/Q matrix complex numbers.   
# More information about Windowing and Hann function:   
# [Wikipedia Hann](https://en.wikipedia.org/wiki/Hann_function#:~:text=The%20Hann%20function%20of%20length,can%20be%20even%20or%20odd.)   
# [Wikipedia Window function](https://en.wikipedia.org/wiki/Window_function)   
# 2. Set max value on the center of the target's mass.   
# The doppler burst is a vector (1 x 32) that marks the center of the target's mass on each I/Q matrix in every time unit (32 time-units).   
# The preprocess sets the maximum value of each I/Q matrix in-place of the values at the center of the target's mass.
# 3. Normalization - Transforming I/Q matrix values to standardized values.   
# Subtraction of the mean and division by the standard deviation.
# 
# **Preprocessing main steps explained:**   
# The DFT converts a signal from its original domain (often time or space) to a representation in the frequency domain and vice versa.   
# The I/Q matrix in this challenge is being converted from the velocity domain to the frequency domain.
# Windows (like "Hann") are used to reduce spectral leakage when performing a Fourier Transform on time data and converting it into the frequency domain.
# 
# The purpose of switching the values of the center of the target's mass in the I/Q matrix with the max value of the matrix is to enhance this part of the matrix and focus the model on this part.
# 
# 
# Finally, the normalization is used for scaling all the I/Q matrices.
# 
# * *Note: the target_type lables are strings ("human" and "animal"), the data_preprocess function replaces them to "human" = 1 and "animal" = 0.*   
#    
# 
# 
# **The FFT Process Explained:**   
# The IQ matrix contains 32 x 128 elements being 128 I/Q time data samples within a single radar processing time frame and 32 consecutive such time units. The X-axis represents the pulse transmission time, also known as “slow-time”. The Y-axis represents the reception time of signals with respect to pulse transmission time divided into 128 equal sized bins, also known as “fast-time”. The reason FFT is performed on the ‘fast time’ axis (i.e. 128 samples) rather than on the ‘slow time’ axis (i.e. 32 samples) is a matter of scale. An underlying assumption for an effective FFT analysis is that ‘acceleration’ during the time frame represented by the data is very low. Given the type of motion presented by humans and animals only the ‘fast time’ axis complies with this assumption.
# Therefore, FFT processing should be applied over the dimension of 128 I/Q time samples to get the frequency content of the returned radar signal at each time frame. A spectrogram (as shown in the samples) is generated by attaching together 32 consecutive FFT results (please note that besides FFT processing some additional manipulation is required to properly display a spectrogram – like taking a LOG of the absolute value and passing a threshold – this is well detailed in the ‘Processing’ section of the competition notebook). Therefore, a spectrogram would eventually provide a display of how the frequency content of the returned signal is changing along time.

# Functions for preprocessing and preprocess function
def fft(iq, axis=0):
	"""
	Computes the log of discrete Fourier Transform (DFT).
		 
	Arguments:
		iq_burst -- {ndarray} -- 'iq_sweep_burst' array
		axis -- {int} -- axis to perform fft in (Default = 0)

	Returns:
		log of DFT on iq_burst array
	"""
	iq = np.log(np.abs(np.fft.fft(hann(iq), axis=axis)))
	return iq


def hann(iq, window=None):
		"""
		Preformes Hann smoothing of 'iq_sweep_burst'.

		Arguments:
			iq {ndarray} -- 'iq_sweep_burst' array
			window -{range} -- range of hann window indices (Default=None)
							 if None the whole column is taken

		Returns:
			Regulazied iq in shape - (window[1] - window[0] - 2, iq.shape[1])
		"""
		if window is None:
				window = [0, len(iq)]

		N = window[1] - window[0] - 1
		n = np.arange(window[0], window[1])
		n = n.reshape(len(n), 1)
		hannCol = 0.5 * (1 - np.cos(2 * np.pi * (n / N)))
		return (hannCol * iq[window[0]:window[1]])[1:-1]


def max_value_on_doppler(iq, doppler_burst):
	"""
	Set max value on I/Q matrix using doppler burst vector. 
		 
	Arguments:
		iq_burst -- {ndarray} -- 'iq_sweep_burst' array
		doppler_burst -- {ndarray} -- 'doppler_burst' array (center of mass)
							 
	Returns:
		I/Q matrix with the max value instead of the original values
		The doppler burst marks the matrix values to change by max value
	"""
	iq_max_value = np.max(iq)
	for i in range(iq.shape[1]):
		if doppler_burst[i]>=len(iq):
			 continue
		iq[doppler_burst[i], i] = iq_max_value
	return iq


def normalize(iq):
	"""
	Calculates normalized values for iq_sweep_burst matrix:
	(vlaue-mean)/std.
	"""
	m = iq.mean()
	s = iq.std()
	return (iq-m)/s


def data_preprocess(data):
	"""
	Preforms data preprocessing.
	Change target_type lables from string to integer:
	'human'  --> 1
	'animal' --> 0

	Arguments:
		data -- {ndarray} -- the data set

	Returns:
		processed data (max values by doppler burst, DFT, normalization)
	"""
	X=[]
	for i in range(len(data['iq_sweep_burst'])):
		iq = fft(data['iq_sweep_burst'][i])
		iq = max_value_on_doppler(iq,data['doppler_burst'][i])
		iq = normalize(iq)
		X.append(iq)

	data['iq_sweep_burst'] = np.array(X)
	if 'target_type' in data:
		data['target_type'][data['target_type'] == 'animal'] = 0
		data['target_type'][data['target_type'] == 'human'] = 1
	return data


# ### **Splitting the Training set**
# 
# The functions below split the training set into Train and Validation sets.
# 
# * Validation Set.   
# The validation set is constructed from segments from geolocation_ids 1 and 4.   
# These geolocations' target_type distributions are balanced.
# To create a fair representation of the test set   
# (one segment form each track) every sixth segment was taken to the validation set.   
# The reason is that approximately 75% of the tracks have less than 6 segments (see the [Descriptive Statistics notebook](https://colab.research.google.com/drive/11Lzihg2vKIbo4KAIIJxW5CRZIncoWgtL?usp=sharing)),   
# it guarantees that most of the segments in the validation set are not from the same track.   
#    
# * Adding "human" segments to the training set.   
# The training set is imbalanced (more animals than humans, see the [Descriptive Statistics notebook](https://colab.research.google.com/drive/11Lzihg2vKIbo4KAIIJxW5CRZIncoWgtL?usp=sharing).   
# One can address this issue by taking segements of humans from the "Auxiliary Experiment Set".   
# The first 3 segments from each track are taken into the training set (or less if the track is shorter than 3 segments).

# Function for splitting the data to training and validation
# and function for selecting samples of segments from the Auxiliary dataset
def split_train_val(data):
	"""
	Split the data to train and validation set.
	The validation set is built from training set segments of 
	geolocation_id 1 and 4. 
	Use the function only after the training set is complete and preprocessed. 

	Arguments:
		data -- {ndarray} -- the data set to split

	Returns:
		iq_sweep_burst ndarray matrices
		target_type vector 
		for training and validation sets
	"""
	idx = ((data['geolocation_id'] == 4) | (data['geolocation_id'] == 1))   & (data['segment_id'] % 6 == 0)
	training_x = data['iq_sweep_burst'][np.logical_not(idx)]
	training_y = data['target_type'][np.logical_not(idx)]
	validation_x = data['iq_sweep_burst'][idx]
	validation_y = data['target_type'][idx]
	return training_x, training_y, validation_x, validation_y

def aux_split(data):
	"""
	Selects segments from the auxilary set for training set.
	Takes the first 3 segments (or less) from each track.

	Arguments:
		data {dataframe} -- the auxilary data

	Returns:
		The auxilary data for the training
	"""
	idx = np.bool_(np.zeros(len(data['track_id'])))
	for track in np.unique(data['track_id']):
		idx |= data['segment_id']==(data['segment_id'][data['track_id'] == track][:3])
	
	for key in data:
		data[key] = data[key][idx]
	return data


# ## **Model**

# ### **CNN Model**
# The model has two convolutional layers, both followed by max-pooling layers.    
# Those layers are followed by 2 fully-connected (dense) layers, activated with a ReLU function and regularized with   
# L2 regularization, followed by a final output layer with a single neuron with a Sigmoid activation function,   
# used for final binary classification.


# Building the model
def create_model(input_shape, init):
	"""
	CNN model.

	Arguments:
		input_shape -- the shape of our input
		init -- the weight initialization

	Returns:
		CNN model    
	"""
	'''
	model = Sequential()
	model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', kernel_initializer = init, bias_regularizer='l2', input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer = init, bias_regularizer='l2'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer = init, bias_regularizer='l2'))
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer = init, bias_regularizer='l2'))
	model.add(Conv2D(32, kernel_size=(1, 1), activation='relu', kernel_initializer = init, bias_regularizer='l2'))
	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(128, kernel_regularizer = 'l2', activation='relu', kernel_initializer = init))
	#model.add(Dense(32, kernel_regularizer = 'l2', activation='relu', kernel_initializer = init))
	#model.add(Dropout(0.2))
	model.add(Dense(1, activation='sigmoid', kernel_initializer = init))
	return model
	'''
	x =  inp(shape=input_shape)
	
	# Initial Layers
	x1 = Conv2D(128, 5, activation="relu", kernel_initializer = init, bias_regularizer='l2', padding='same')(x)
	x2 = MaxPooling2D(pool_size=(2, 2), strides=2)(x1)
	x3 = BatchNormalization()(x2)
	
	# Dense Block 1
	x3_input = x3
	x4 = Conv2D(256, 1, activation="relu", kernel_initializer = init, bias_regularizer='l2', padding='same')(x3_input)
	x5 = BatchNormalization()(x4)
	x6 = Conv2D(256, 3, activation="relu", kernel_initializer = init, bias_regularizer='l2', padding='same')(x5)
	x7 = BatchNormalization()(x6)
	x8 = Conv2D(256, 1, activation="relu", kernel_initializer = init, bias_regularizer='l2', padding='same')(x7)
	x9 = BatchNormalization()(x8)
	x10 = Conv2D(256, 3, activation="relu", kernel_initializer = init, bias_regularizer='l2', padding='same')(x9)
	x11 = BatchNormalization()(x10)
	
	# Transition Layer 1
	x11_input = Concatenate()([x3,x11])
	x12 = Conv2D(512,1, activation="relu", kernel_initializer = init, bias_regularizer='l2', padding='same')(x11_input)
	x13 = MaxPooling2D(pool_size=(2, 2), strides=2)(x12)

	# Dense Block 2
	x14 = Conv2D(512, 1, activation="relu", kernel_initializer = init, bias_regularizer='l2', padding='same')(x13)
	x15 = BatchNormalization()(x14)
	x16 = Conv2D(512, 3, activation="relu", kernel_initializer = init, bias_regularizer='l2', padding='same')(x15)
	x17 = BatchNormalization()(x16)
	x18 = Conv2D(512, 1, activation="relu", kernel_initializer = init, bias_regularizer='l2', padding='same')(x17)
	x19 = BatchNormalization()(x18)
	x20 = Conv2D(512, 3, activation="relu", kernel_initializer = init, bias_regularizer='l2', padding='same')(x19)
	x21 = BatchNormalization()(x20)
	
	# Transition Layer 2
	x21_input = Concatenate()([x13,x21])
	x22 = Conv2D(1024, 1, activation="relu", kernel_initializer = init, bias_regularizer='l2', padding='same')(x21_input)
	x23 = BatchNormalization()(x22)

	# Final Layer
	x24 = GlobalAveragePooling2D()(x23)
	x25 = Flatten()(x24)
	x26 = Dense(1, activation='sigmoid', kernel_initializer = init)(x25)

	model = Model(inputs=x, outputs=x26)

	return model

# ### **Evaluation and Visualization of Model's results**

# Function for calculating the final ROC-AUC score and plot the ROC curve,
# used in the "Results" section
def stats(pred, actual):
	"""
	Computes the model ROC-AUC score and plots the ROC curve.

	Arguments:
		pred -- {ndarray} -- model's probability predictions
		actual -- the true lables

	Returns:
		ROC curve graph and ROC-AUC score
	"""
	plt.figure(figsize=(20, 10))
	fpr1, tpr1, _ = roc_curve(actual[0], pred[0])
	fpr2, tpr2, _ = roc_curve(actual[1], pred[1])
	roc_auc = [auc(fpr1, tpr1), auc(fpr2, tpr2)]
	lw = 2
	plt.plot(fpr1, tpr1, lw=lw, label='Training set (ROC-AUC = %0.2f)' % roc_auc[0])
	plt.plot(fpr2, tpr2, lw=lw, label='Validation set (ROC-AUC = %0.2f)' % roc_auc[1])
	plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--', label = 'Random guess')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate', fontsize=18)
	plt.ylabel('True Positive Rate', fontsize=18)
	plt.title('Training set vs. Validation set ROC curves')
	plt.legend(loc="lower right", prop = {'size': 20})
	plt.show()


# ## **Training The Model**

# Loading and preparing the data

# Loading Auxiliary Experiment set - can take a few minutes
experiment_auxiliary = '/home/vaibhawvipul/Documents/vaibhawvipul/datasets/mafat/MAFAT RADAR Challenge - Auxiliary Experiment Set V2'
experiment_auxiliary_df = load_data(experiment_auxiliary)



# Taking sample from the Auxiliary Experiment set
train_aux = aux_split(experiment_auxiliary_df)


# The function append_dict is for concatenating the training set 
# with the Auxiliary data set segments

def append_dict(dict1, dict2):
	for key in dict1:
		dict1[key] = np.concatenate([dict1[key], dict2[key]], axis=0)
	return dict1


# Training set
train_path = '/home/vaibhawvipul/Documents/vaibhawvipul/datasets/mafat/MAFAT RADAR Challenge - Training Set V1'
training_df = load_data(train_path)

# Adding segments from the experiment auxiliary set to the training set
train_df = append_dict(training_df, train_aux)

# Preprocessing and split the data to training and validation
train_df = data_preprocess(train_df.copy())
#df = pd.DataFrame.from_dict(train_df, orient="index")
#df = df.transpose()
#df = df[['iq_sweep_burst','target_type']]
x = train_df["iq_sweep_burst"]
y = train_df["target_type"].astype(int)

orig_shape = (-1,x.shape[1],x.shape[2])

x = x.reshape(x.shape[0],x.shape[1]*x.shape[2])

smote = SMOTE(random_state=0)
X_resampled, y_resampled = smote.fit_sample(x, y)

train_x, _, train_y, _ = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=1)
#train_x, train_y, val_x, val_y = split_train_val(train_df)
#val_y =  val_y.astype(int)
train_y = train_y.astype(int)

train_x = train_x.reshape(orig_shape)
train_x = train_x.reshape(list(train_x.shape)+[1])

_, _, val_x, val_y = split_train_val(train_df)
val_y =  val_y.astype(int)
val_x = val_x.reshape(list(val_x.shape)+[1])


# Public test set - loading and preprocessing
test_path = '/home/vaibhawvipul/Documents/vaibhawvipul/datasets/mafat/MAFAT RADAR Challenge - Public Test Set V1'
test_df = load_data(test_path)
test_df = data_preprocess(test_df.copy())
test_x = test_df['iq_sweep_burst']
test_x = test_x.reshape(list(test_x.shape)+[1])


# Model configuration:
batch_size = 32
img_width, img_height = 126, 32
loss_function = BinaryCrossentropy()
no_epochs = 10
optimizer = Adam(learning_rate = 0.0001)
input_shape = (img_width, img_height, 1)

init = tf.keras.initializers.glorot_normal(seed = 0)

# Creating and running the model
model = create_model(input_shape, init)  
model.compile(loss=loss_function, optimizer=optimizer, metrics=[AUC(), 'accuracy'])

# **Model Architecture**   
#    
# ![](https://drive.google.com/uc?export=view&id=1wsJBHbghEPGT0s1QQG6BHl7MS3Yo0o4i)


model.summary()


# Model fit
history = model.fit(train_x, train_y, batch_size = batch_size, epochs = no_epochs, 
										validation_data = (val_x, val_y))


# #### **Results**
# Submissions are evaluated on the area under the Receiver Operating Characteristic Curve ([ROC AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic))   
# on the predicted probabilities, as calculated by [roc_auc_score in scikit-learn (v 0.23.1)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).
# 

# Plot ROC curve and show ROC-AUC results of the training and validation sets. 
pred = [model.predict(train_x), model.predict(val_x)]
actual = [train_y, val_y]
stats(pred, actual)


# Creating DataFrame with the probability prediction for each segment
submission =  pd.DataFrame()
submission['segment_id'] = test_df['segment_id']
submission['prediction'] = model.predict(test_x)
submission['prediction'] = submission['prediction'].astype('float')

# Save submission
submission.to_csv('submission.csv', index=False)