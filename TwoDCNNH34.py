import pickle
import pandas as pd
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
import collections
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
from sklearn.model_selection import StratifiedKFold
from collections import Counter


def main(path):
    batch_size = 256
    epochs = 50
    seed = 7
    np.random.seed(seed)

    df = pd.read_pickle(path)

    X_cancer_samples = df.loc[df['label'] != 0].values[:,0:-1]
    X_normal_samples = df.loc[df['label'] == 0].values[:,0:-1]

    name_cancer_samples = df.loc[df['label'] != 0].values[:,-1]
    name_normal_samples = df.loc[df['label'] == 0].values[:,-1]

    X_cancer_samples_34 = np.concatenate((X_cancer_samples,X_normal_samples))
    X_names = np.concatenate((name_cancer_samples,name_normal_samples))
    X_cancer_samples_mat = np.concatenate((X_cancer_samples_34,np.zeros((len(X_cancer_samples_34),21))),axis=1)
    X_cancer_samples_mat = np.reshape(X_cancer_samples_mat, (-1, 71, 100))


    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cvscores = []
    cv_yscores = []
    Y_test =[]
    instance_mean = []

    input_Xs = X_cancer_samples_mat
    y_s = X_names

    img_rows, img_cols = len(input_Xs[0][0]), len(input_Xs[0])
    num_classes = len(set(y_s))

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y_s)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    for j in range(10):
        i = 0

        for train, test in kfold.split(input_Xs, y_s):   # input_Xs in normal case and shuffled should be shuffled_Xs

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
            # tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)
            tower_1 = MaxPooling2D(1, 2)(tower_1)
            tower_1 = Flatten()(tower_1)

            tower_2 = Conv2D(32, (100, 1), activation='relu')(input_img)
            # tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)
            tower_2 = MaxPooling2D(1, 2)(tower_2)
            tower_2 = Flatten()(tower_2)


            output = keras.layers.concatenate([tower_1, tower_2], axis=1)

            out1 = Dense(128, activation='relu')(output)
            last_layer = Dense(num_classes, activation='softmax')(out1)

            model = Model(input=[input_img], output=last_layer)
            model.output_shape

            model.compile(loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=['categorical_accuracy'])
            callbacks = [EarlyStopping(monitor='categorical_accuracy', patience=3, verbose=0)]
            if i==0:
                model.summary()
                i = i +1
            history = model.fit(input_Xs[train], onehot_encoded[train],
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=0, callbacks=callbacks, validation_data=(input_Xs[test], onehot_encoded[test]))
            scores = model.evaluate(input_Xs[test], onehot_encoded[test], verbose=0)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            cvscores.append(scores[1] * 100)

        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        # cv_yscores = np.concatenate(cv_yscores)
        # Y_test = np.concatenate(Y_test)
        instance_mean.append(np.mean(cvscores))
    
    return instance_mean


def saveresult(x, name):
    x = np.array(x)
    np.savetxt('noise_result_%s.csv' % name, x, delimiter=',')
    
if __name__ == '__main__':
    paths = ['E:/Google Drive (CMU)/DL project/code/tcga_data/noise_11093x7080/11093x7080_p3_noisy_10.pkl',
            'E:/Google Drive (CMU)/DL project/code/tcga_data/noise_11093x7080/11093x7080_p3_noisy_50.pkl',
            'E:/Google Drive (CMU)/DL project/code/tcga_data/noise_11093x708011093x7080_p3_noisy_100.pkl',
            'E:/Google Drive (CMU)/DL project/code/tcga_data/noise_11093x7080/11093x7080_p3_noisy_200.pkl',
            'E:/Google Drive (CMU)/DL project/code/tcga_data/noise_11093x7080/11093x7080_p3_noisy_500.pkl']
    means = []
    for path in paths:
        istance_mean = main(path)
        means.append(istance_mean)

    saveresult(means, '2dcnnv33')