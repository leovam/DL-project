import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import pickle
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
import collections
import matplotlib.pyplot as plt
import pandas as pd
import keras.layers as klayers
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
from sklearn.model_selection import StratifiedKFold
import os
import sys
from keras.models import load_model
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time

def OneD33_model(train, test, input_Xs, y_s, r, k, e):
	batch_size = 256
	epochs = e
	SAVE_DIR = 'models'
	if not os.path.isdir(SAVE_DIR):
		os.makedirs(SAVE_DIR)
	MODEL_SAVE_PATH_ONEDCNN = os.path.join(SAVE_DIR, '1dcnn33_{}_{}.h5'.format(r, k))

	img_rows, img_cols = len(input_Xs[0][0]), len(input_Xs[0])
	input_Xs = input_Xs.reshape(input_Xs.shape[0], img_rows, img_cols, 1)
	input_shape = (img_rows, img_cols, 1)
	input_Xs = input_Xs.astype('float32')

	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(y_s)
	# binary encode
	onehot_encoder = OneHotEncoder(sparse=False)
	integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
	onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
	num_classes = len(onehot_encoded[0])

	model = Sequential()
	## *********** First layer Conv
	model.add(Conv2D(32, kernel_size=(1, 71), strides=(1, 1),
							input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(1, 2))
	## ********* Classification layer
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy',
						optimizer='adam',
						metrics=['categorical_accuracy'])
	callbacks = [EarlyStopping(monitor='categorical_accuracy', patience=3, verbose=0)]

	if r == 0 and k == 0:
		model.summary()

	history = model.fit(input_Xs[train], onehot_encoded[train],
								batch_size=batch_size,
								epochs=epochs,
								verbose=0, callbacks=callbacks, validation_data=(input_Xs[test], onehot_encoded[test]))
	scores = model.evaluate(input_Xs[test], onehot_encoded[test], verbose=0)
	# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	model.save(MODEL_SAVE_PATH_ONEDCNN)

	return scores[1]

#Obtain prediction for trainnings
def OneD33_preidct(indexes, input_Xs, y_s, r, k):
	batch_size = 256
	SAVE_DIR = 'models'
	if not os.path.isdir(SAVE_DIR):
		os.makedirs(SAVE_DIR)
	MODEL_SAVE_PATH_ONEDCNN = os.path.join(SAVE_DIR, '1dcnn33_{}_{}.h5'.format(r, k))

	model = load_model(MODEL_SAVE_PATH_ONEDCNN)

	img_rows, img_cols = len(input_Xs[0][0]), len(input_Xs[0])
	input_Xs = input_Xs.reshape(input_Xs.shape[0], img_rows, img_cols, 1)
	input_shape = (img_rows, img_cols, 1)
	input_Xs = input_Xs.astype('float32')

	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(y_s)
	# binary encode
	onehot_encoder = OneHotEncoder(sparse=False)
	integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
	onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
	num_classes = len(onehot_encoded[0])

	pred = model.predict(input_Xs[indexes], verbose=0)
	
	return pred

def TwoD33v_model(train, test, input_Xs, y_s, r, k, e):
	batch_size = 256
	epochs = e
	SAVE_DIR = 'models'
	if not os.path.isdir(SAVE_DIR):
		os.makedirs(SAVE_DIR)
	MODEL_SAVE_PATH_TWODCNN = os.path.join(SAVE_DIR, '2dvcnn33_{}_{}.h5'.format(r, k))

	img_rows, img_cols = len(input_Xs[0][0]), len(input_Xs[0])
	input_Xs = input_Xs.reshape(input_Xs.shape[0], img_rows, img_cols, 1)
	input_shape = (img_rows, img_cols, 1)
	input_Xs = input_Xs.astype('float32')

	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(y_s)
	# binary encode
	onehot_encoder = OneHotEncoder(sparse=False)
	integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
	onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
	num_classes = len(onehot_encoded[0])

	model = Sequential()
	## *********** First layer Conv
	model.add(Conv2D(32, kernel_size=(10, 10), strides=(2, 2),
					input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(2, 2))
	## ********* Classification layer
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy',
				optimizer='adam',
				metrics=['categorical_accuracy'])
	callbacks = [EarlyStopping(monitor='categorical_accuracy', patience=3, verbose=0)]

	if r == 0 and k == 0:
		model.summary()

	history = model.fit(input_Xs[train], onehot_encoded[train],
								batch_size=batch_size,
								epochs=epochs,
								verbose=0, callbacks=callbacks, validation_data=(input_Xs[test], onehot_encoded[test]))
	scores = model.evaluate(input_Xs[test], onehot_encoded[test], verbose=0)
	# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	model.save(MODEL_SAVE_PATH_TWODCNN)

	return scores[1]

#Obtain prediction for trainnings
def TwoD33v_predict(indexes, input_Xs, y_s, r, k):
	batch_size = 256
	SAVE_DIR = 'models'
	if not os.path.isdir(SAVE_DIR):
		os.makedirs(SAVE_DIR)
	MODEL_SAVE_PATH_TWODCNN = os.path.join(SAVE_DIR, '2dvcnn33_{}_{}.h5'.format(r, k))

	model = load_model(MODEL_SAVE_PATH_TWODCNN)

	img_rows, img_cols = len(input_Xs[0][0]), len(input_Xs[0])
	input_Xs = input_Xs.reshape(input_Xs.shape[0], img_rows, img_cols, 1)
	input_shape = (img_rows, img_cols, 1)
	input_Xs = input_Xs.astype('float32')

	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(y_s)
	# binary encode
	onehot_encoder = OneHotEncoder(sparse=False)
	integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
	onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
	num_classes = len(onehot_encoded[0])

	pred = model.predict(input_Xs[indexes], verbose=0)
	
	return pred

def TwoD33h_model(train, test, input_Xs, y_s, r, k, e):
	batch_size = 256
	epochs = e
	SAVE_DIR = 'models'
	if not os.path.isdir(SAVE_DIR):
		os.makedirs(SAVE_DIR)
	MODEL_SAVE_PATH_TWODCNN = os.path.join(SAVE_DIR, '2dhcnn33_{}_{}.h5'.format(r, k))

	img_rows, img_cols = len(input_Xs[0][0]), len(input_Xs[0])

	input_Xs = input_Xs.reshape(input_Xs.shape[0], img_rows, img_cols, 1)
	input_shape = (img_rows, img_cols, 1)
	input_Xs = input_Xs.astype('float32')
	input_img = Input(input_shape)
	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(y_s)
	# binary encode
	onehot_encoder = OneHotEncoder(sparse=False)
	integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
	onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
	num_classes = len(onehot_encoded[0])

	tower_1 = Conv2D(32, (1, 71), activation='relu')(input_img)
	tower_1 = MaxPooling2D(1, 2)(tower_1)
	tower_1 = Flatten()(tower_1)

	tower_2 = Conv2D(32, (100, 1), activation='relu')(input_img)
	tower_2 = MaxPooling2D(1, 2)(tower_2)
	tower_2 = Flatten()(tower_2)

	output = klayers.concatenate([tower_1, tower_2], axis=1)
	out1 = Dense(128, activation='relu')(output)
	last_layer = Dense(num_classes, activation='softmax')(out1)
	model = Model(input=[input_img], output=last_layer)
	model.output_shape

	model.compile(loss='categorical_crossentropy',
						optimizer='adam',
						metrics=['categorical_accuracy'])
	callbacks = [EarlyStopping(monitor='categorical_accuracy', patience=3, verbose=0)]

	if r == 0 and k == 0:
		model.summary()

	history = model.fit(input_Xs[train], onehot_encoded[train],
								batch_size=batch_size,
								epochs=epochs,
								verbose=0, callbacks=callbacks, validation_data=(input_Xs[test], onehot_encoded[test]))
	scores = model.evaluate(input_Xs[test], onehot_encoded[test], verbose=0)

	model.save(MODEL_SAVE_PATH_TWODCNN)

	return scores[1]

#Obtain predictions for training
def TwoD33h_predict(indexes, input_Xs, y_s, r, k):
	batch_size = 256
	SAVE_DIR = 'models'
	if not os.path.isdir(SAVE_DIR):
		os.makedirs(SAVE_DIR)
	MODEL_SAVE_PATH_TWODCNN = os.path.join(SAVE_DIR, '2dhcnn33_{}_{}.h5'.format(r, k))

	model = load_model(MODEL_SAVE_PATH_TWODCNN)

	img_rows, img_cols = len(input_Xs[0][0]), len(input_Xs[0])

	input_Xs = input_Xs.reshape(input_Xs.shape[0], img_rows, img_cols, 1)
	input_shape = (img_rows, img_cols, 1)
	input_Xs = input_Xs.astype('float32')
	input_img = Input(input_shape)
	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(y_s)
	# binary encode
	onehot_encoder = OneHotEncoder(sparse=False)
	integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
	onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
	num_classes = len(onehot_encoded[0])

	pred = model.predict(input_Xs[indexes], verbose=0)

	return pred

def fit_stack(train, test, input_Xs, y_s, r, k):
	pred1d = OneD33_preidct(train, input_Xs, y_s, r, k)
	pred2dh = TwoD33h_predict(train, input_Xs, y_s, r, k)
	pred2dv = TwoD33v_predict(train, input_Xs, y_s, r, k)

	stacked_x = np.dstack((pred1d, pred2dh, pred2dv))
	stacked_x = stacked_x.reshape(stacked_x.shape[0], stacked_x.shape[1] * stacked_x.shape[2])

	model = LogisticRegression()

	model.fit(stacked_x, y_s[train])
	train_score = model.score(stacked_x, y_s[train])
	print("train score: {}".format(train_score))

	SAVE_DIR = 'modelsens33'
	if not os.path.isdir(SAVE_DIR):
		os.makedirs(SAVE_DIR)
	MODEL_SAVE_PATH_LINEARREG = os.path.join(SAVE_DIR, 'lreg_{}_{}.sav'.format(r, k))

	pickle.dump(model, open(MODEL_SAVE_PATH_LINEARREG, 'wb'))
	model = pickle.load(open(MODEL_SAVE_PATH_LINEARREG, 'rb'))

	test1d = OneD33_preidct(test, input_Xs, y_s, r, k)
	test2dh = TwoD33h_predict(test, input_Xs, y_s, r, k)
	test2dv = TwoD33v_predict(test, input_Xs, y_s, r, k)

	stacked_xtest = np.dstack((test1d, test2dh, test2dv))
	stacked_xtest = stacked_xtest.reshape(stacked_xtest.shape[0], stacked_xtest.shape[1] * stacked_xtest.shape[2])
	test_score = model.score(stacked_xtest, y_s[test])
	print("test score: {}".format(test_score))

	return train_score, test_score

def score_stack_only(train, test, input_Xs, y_s, r, k):
	SAVE_DIR = 'modelsens33'
	if not os.path.isdir(SAVE_DIR):
		os.makedirs(SAVE_DIR)
	MODEL_SAVE_PATH_LINEARREG = os.path.join(SAVE_DIR, 'lreg_{}_{}.sav'.format(r, k))

	model = pickle.load(open(MODEL_SAVE_PATH_LINEARREG, 'rb'))

	test1d = OneD33_preidct(test, input_Xs, y_s, r, k)
	test2dh = TwoD33h_predict(test, input_Xs, y_s, r, k)
	test2dv = TwoD33v_predict(test, input_Xs, y_s, r, k)

	stacked_xtest = np.dstack((test1d, test2dh, test2dv))
	stacked_xtest = stacked_xtest.reshape(stacked_xtest.shape[0], stacked_xtest.shape[1] * stacked_xtest.shape[2])
	test_score = model.score(stacked_xtest, y_s[test])
	print("test score: {}".format(test_score))

	return 0, test_score


def load_data(path):
	df = pd.read_pickle(path)

	X_cancer_samples = df.loc[df['label'] != 0].values[:,0:-1]
	name_cancer_samples = df.loc[df['label'] != 0].values[:,-1]

	X_cancer_samples_mat = np.concatenate((X_cancer_samples,np.zeros((len(X_cancer_samples),21))),axis=1)
	X_cancer_samples_mat = np.reshape(X_cancer_samples_mat, (-1, 71, 100))

	input_Xs = X_cancer_samples_mat
	y_s = name_cancer_samples

	return input_Xs, y_s

def main():
	REPS = 10
	SEED = 7
	USE_MODEL = 0 #whether load pre-trained CNNs
	EPOCH = 50
	if len(sys.argv) > 2:
		if sys.argv[2] == '-u':
			USE_MODEL = 1
	np.random.seed(SEED)

	ens_train_acc_all = []
	ens_test_acc_all = []

	#path = 'preprocessed_11093x7080_p3.pkl'
	path = sys.argv[1]
	input_Xs, y_s = load_data(path)
	kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


	for r in range(REPS):
		start_time = time.time()
		k = 0
		ens_train_acc, ens_test_acc = [], []

		for train, test in kfold.split(input_Xs, y_s):
			if not USE_MODEL: #Train the CNNs and save model
				s1 = OneD33_model(train, test, input_Xs, y_s, r, k, EPOCH)
				print("s1 {}".format(s1))
				s2h = TwoD33h_model(train, test, input_Xs, y_s, r, k, EPOCH)
				print("s2h {}".format(s2h))
				s2v = TwoD33v_model(train, test, input_Xs, y_s, r, k, EPOCH)
				print("s2v {}".format(s2v))

				#Ensemble
				ens_train, ens_test = fit_stack(train, test, input_Xs, y_s, r, k)
			else:
				ens_train, ens_test = score_stack_only(train, test, input_Xs, y_s, r, k)

			ens_train_acc.append(ens_train)
			ens_test_acc.append(ens_test)
			k += 1

		avg_train, avg_test = sum(ens_train_acc) / len(ens_train_acc), sum(ens_test_acc) / len(ens_test_acc)
		ens_train_acc_all.append(avg_train)
		ens_test_acc_all.append(avg_test)

		print(ens_train_acc_all)
		print(ens_test_acc_all)
		print(time.time() - start_time)

	return


if __name__ == '__main__':
	main()
