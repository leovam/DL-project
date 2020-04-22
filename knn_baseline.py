import numpy as np
import pandas as pd
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import time
import sys

df = pd.read_pickle('preprocessed_11093x7080_p3.pkl')
noise_level = int(sys.argv[1])
df_noise = pd.read_pickle('../../Downloads/11093x7080_p3_noisy_{}.pkl'.format(noise_level))
#df_noise = pd.read_pickle('preprocessed_11093x7080_p3.pkl')
print('Num of entries (samples):', df.shape[0])
print('Num of features (genes):', df.shape[1] - 1)
print('Num of classes:', len(list(df.iloc[:, -1].unique())))
print("Number of normal samples:", df.loc[df['label'] == 0].shape[0])
print("Number of tumor samples:", df.loc[df['label'] != 0].shape[0])

print("selecting all vs cancer only")
#Get label and features for only cancer samples
label_cancer_samples = df.loc[df['label'] != 0].values[:, -1]
X_cancer_samples = df.loc[df['label'] != 0].values[:, 0:-1]
X_cancer_samples = np.concatenate((X_cancer_samples, np.zeros((len(X_cancer_samples),21))), axis=1)
X_cancer_samples_noise = df_noise.loc[df['label'] != 0].values[:, 0:-1]
X_cancer_samples_noise = np.concatenate((X_cancer_samples_noise, np.zeros((len(X_cancer_samples_noise),21))), axis=1)

#Get label and feature for all samples (including normal)
label_all_samples = df.values[:, -1]
X_all_samples = df.values[:, 0:-1]
X_all_samples = np.concatenate((X_all_samples, np.zeros((len(X_all_samples),21))), axis=1)
X_all_samples_noise = df_noise.values[:, 0:-1]
X_all_samples_noise = np.concatenate((X_all_samples_noise, np.zeros((len(X_all_samples_noise),21))), axis=1)

results_all, results_can = [], []
rf_all, rf_can = [], []


#Generate training, testing, and whole data
test_size = 1000 #Approximately 10% of data
np.random.seed(1234)
n_num = int(sys.argv[2])

for r in range(10): #10 reps
	start_time = time.time()
	test_ind_cancer = np.random.choice(label_cancer_samples.shape[0], test_size, replace = False)
	test_ind_all = np.random.choice(label_all_samples.shape[0], test_size, replace = False)
	train_ind_cancer = np.delete(np.arange(label_cancer_samples.shape[0]), test_ind_cancer)
	train_ind_all = np.delete(np.arange(label_all_samples.shape[0]), test_ind_all)

	test_X_cancer, train_X_cancer = X_cancer_samples_noise[test_ind_cancer], X_cancer_samples[train_ind_cancer]
	test_X_all, train_X_all = X_all_samples_noise[test_ind_all], X_all_samples[train_ind_all]
	test_label_cancer, train_label_cancer = label_cancer_samples[test_ind_cancer], label_cancer_samples[train_ind_cancer]
	test_label_all, train_label_all = label_all_samples[test_ind_all], label_all_samples[train_ind_all]


	
	print("num neighbors = {}".format(n_num))
	print("setting trains for all samples")
	#neigh = KNeighborsClassifier(n_neighbors = n_num, metric = 'correlation')
	neigh = KNeighborsClassifier(n_neighbors = n_num, metric = 'minkowski', p=2)
	neigh.fit(train_X_all, train_label_all)
	print("scoring for all samples")
	s_all = neigh.score(test_X_all, test_label_all)
	print(s_all)

	print("setting trains for cancer samples")
	#neigh = KNeighborsClassifier(n_neighbors = n_num, metric = 'correlation')
	neigh = KNeighborsClassifier(n_neighbors = n_num, metric = 'minkowski', p=2)
	neigh.fit(train_X_cancer, train_label_cancer)
	print("scoring for cancer samples")
	s_can = neigh.score(test_X_cancer, test_label_cancer)
	print(s_can)

	results_all.append(s_all)
	results_can.append(s_can)

	print("time used: {}".format(time.time() - start_time))

print("finished.", noise_level)
print(results_all)
print(results_can)
