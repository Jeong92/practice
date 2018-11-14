# library import
from typing import List, Any

import scipy.io as sio  # load .mat file
import os  # directory setting
import numpy as np
from keras import layers, models, optimizers
from sklearn.model_selection import StratifiedKFold, train_test_split


# load data for classification using LSTM
def Data():
    # define erp directory
    erp_dir = 'C:\\Users\\user\\PycharmProjects\\DeepLearning\\eeg_classification\\'  # eeg data directory
    print('erp directory: \n', erp_dir, end='\n')

    # load erp file
    erp_file = 'erp_wordretrieval_N69.mat'
    erp_data = sio.loadmat(erp_dir + erp_file)
    erp_raw = erp_data['erp_x']
    n_sample, n_classes = np.shape(erp_raw)
    print('Finished ERP Data per %s classes from %s participants\n' % (n_classes, n_sample))

    # split train-test data for resampling(N=5)
    n_resample = 5
    folds = []
    for k in range(0, n_resample):
        folds.append(train_test_split([x for x in range(n_sample)], shuffle=True, test_size=0.2))

    return erp_raw, n_classes, n_sample, folds


# LSTM model
class RNN_LSTM(models.Model):
    def __init__(self):
        x = layers.Input(shape=(301, 62))
        h = layers.LSTM(62, dropout=0.4, recurrent_dropout=0.4)(x)
        y = layers.Dense(1, activation='sigmoid')(h)
        super().__init__(x, y)

        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9)

        # try using different optimizers and different optimizer config
        self.compile(loss='mse',
                     optimizer='sgd', metrics=['accuracy'])

# load eeg files
def main():
    erp_raw, n_classes, n_sample, folds = Data()

    # for loop
    for j, (train_idx, test_idx) in enumerate(folds):
        train_subj = erp_raw[train_idx]
        test_subj = erp_raw[test_idx]
        # erp sorting
        train_x = np.concatenate((train_subj[:][0], train_subj[:][1]), axis=0)



        train_x, train_y = train_subj[0], train_subj[1]

        model = RNN_LSTM()

