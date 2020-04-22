import numpy as np
import pandas as pd

df = pd.read_pickle('preprocessed_11093x7080_p3.pkl')
feature_num = df.shape[1] - 1
sample_num = df.shape[0]

col_avg = df.mean(axis = 0)

SEED = 1234
np.random.seed(SEED)

def add_noise(k):
	new_df = df.copy()
	noise = [np.random.normal(0, k * col_avg[i], sample_num).reshape(sample_num, 1) for i in range(feature_num)]
	noise_arr = np.hstack(noise)
	label_col_noise = np.zeros((sample_num, 1)) #no noise on label col
	noise_arr = np.hstack([noise_arr, label_col_noise])

	new_df = new_df + noise_arr
	new_df[new_df < 0] = 0
	new_df['label'] = new_df['label'].astype('int32') 
	return new_df


noise_levels = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]

for k in noise_levels:
	print("generating noise for k = {}".format(k))
	noisy = add_noise(k)
	save_path = "./11093x7080_p3_noisy_{}.pkl".format(int(k*100))
	noisy.to_pickle(save_path)
