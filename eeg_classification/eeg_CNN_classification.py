# library import
import scipy.io as sio  # load .mat file
import os  # directory setting
import numpy as np
from keras import models, layers
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize


# CNN model
class CNN(models.Sequential):
    def __init__(self, input_shape, num_classes):
        super().__init__()

        self.add(layers.Conv2D(32, kernel_size=(3, 3),
                               activation='relu',
                               input_shape=input_shape))
        self.add(layers.Conv2D(63, (3, 3), activation='relu'))
        self.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.add(layers.Dropout(0.25))
        self.add(layers.Flatten())
        self.add(layers.Dense(128, activation='relu'))
        self.add(layers.Dropout(0.5))
        self.add(layers.Dense(num_classes, activation='softmax'))

        self.compile(loss='mse',
                     optimizer='sgd',
                     metrics=['accuracy'])


# load data for classification using CNN
class Data:
    def __init__(self):
        # split data by condition
        eeg_raw = eeg_data['EEG_data']

        # eeg data format = # of trial x time point(sample) x channel(feature)
        eeg_target = np.transpose(eeg_raw[0, 0], (2, 0, 1))
        eeg_lure = np.transpose(eeg_raw[0, 1], (2, 0, 1))
        eeg_filler = np.transpose(eeg_raw[0, 2], (2, 0, 1))

        # concatenate lure and filler to non-target trial
        eeg_nontarget = np.append(eeg_lure, eeg_filler, axis=0)  # np.append 사용 방법

        # define eeg_x and eeg_y variable
        num_trial = eeg_target.shape[0]  # number of trial per condition
        self.eeg_x = np.append(eeg_target, eeg_nontarget[np.random.choice(eeg_nontarget.shape[0], num_trial), :, :],
                               axis=0)
        self.eeg_y = np.append(np.zeros(num_trial)+1, np.zeros(num_trial)+2, axis=0)
        self.num_classes = 2
        self.input_shape = (62, 701, 1)

        # normalize eeg signal
        for i in range(self.eeg_x.shape[0]):
            self.eeg_x[i] = normalize(self.eeg_x[i], axis=0)
        self.eeg_x = self.eeg_x.reshape(self.eeg_x.shape[0], 62, 701, 1)

    # load eeg data
eeg_dir = 'C:\\Users\\user\\PycharmProjects\\DeepLearning\\eeg_classification\\raw_eeg' # eeg data directory
print('eeg directory: \n', eeg_dir, end='\n')

eeg_list = os.listdir(eeg_dir)  # os.listdir(dir): showing file list in the directory
print('eeg list: \n', eeg_list, end='\n')

# repetition
for subj_idx in range(len(eeg_list)):

    eeg_file = eeg_dir+'\\'+eeg_list[subj_idx]
    eeg_data = sio.loadmat(eeg_file)  # sio.loadmat=.mat: read.mat file
    print('Load eeg data from %s/%s of participants' % (subj_idx+1, len(eeg_list)))

    # load eeg file
    data = Data()

    # preparation data for k-fold cross validation
    def load_data_kfold(k):
        folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=1).split(data.eeg_x, data.eeg_y))
        return folds, data.eeg_x, data.eeg_y

    k = 5
    folds, x_train, y_train = load_data_kfold(k)

    for j, (train_idx, val_idx) in enumerate(folds):
        print('\nFold ', j)
        x_train_cv = x_train[train_idx]
        y_train_cv = y_train[train_idx]
        x_valid_cv = x_train[val_idx]
        y_valid_cv = y_train[val_idx]

        model = CNN(data.input_shape, data.num_classes)
        print('Training stage')
        print('==================')
        model.fit(x_train_cv, y_train_cv,
                  batch_size=8,
                  epochs=10,
                  validation_data=(x_valid_cv, y_valid_cv))
        score, acc = model.evaluate(x_valid_cv, y_valid_cv, batch_size=8)
        print('Test performance: accuracy={0}, loss={1}'.format(acc, score))