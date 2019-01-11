# library import
import scipy.io as sio  # load .mat file
import numpy as np
from keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder


# load data for classification using CNN
def Data():
    # define erp directory
    erp_dir = 'C:\\Users\\user\\PycharmProjects\\DeepLearning\\eeg_classification\\'  # eeg data directory
    print('erp directory: \n', erp_dir, end='\n')

    # load erp file
    erp_file = 'erp_wordretrieval_N69.mat'
    erp_data = sio.loadmat(erp_dir + erp_file)
    erp_raw = erp_data['erp_x']
    n_sample, n_classes = np.shape(erp_raw)[0], np.shape(erp_raw)[3]
    print('Finished ERP Data per %s classes from %s participants\n' % (n_classes, n_sample))

    # split train-test data for resampling(N=5)
    n_resample = 5
    folds = []
    for k in range(0, n_resample):
        folds.append(train_test_split([x for x in range(n_sample)], shuffle=True, test_size=0.2))

    return erp_raw, n_classes, n_sample, folds


# CNN model
class CNN_model(models.Model):
    def __init__(self):
        x = layers.Input(shape=(301, 62))  # input layer(channel last)
        h = layers.Conv1D(kernel_size=(20), filters=5, activation='relu')(x)
        h = layers.MaxPooling1D(pool_size=2, strides=2)(h)
        h = layers.Conv1D(kernel_size=(10), filters=10, activation='relu')(h)
        h = layers.MaxPooling1D(pool_size=2, strides=2)(h)
        h = layers.Conv1D(kernel_size=(10), filters=10, activation='relu')(h)
        h = layers.MaxPooling1D(pool_size=2, strides=2)(h)
        h = layers.Conv1D(kernel_size=(5), filters=15, activation='relu')(h)
        h = layers.MaxPooling1D(pool_size=2, strides=2)(h)
        h = layers.Flatten()(h)
        h = layers.Dense(20, activation='relu')(h)
        h = layers.Dropout(0.5)(h)
        h = layers.Dense(10, activation='relu')(h)
        h = layers.Dropout(0.5)(h)
        y = layers.Dense(2, activation='softmax')(h)
        super().__init__(x, y)

        adam = optimizers.adam(lr=0.0001)

        # try using different optimizers and different optimizer config
        self.compile(loss='binary_crossentropy',
                     optimizer='adam', metrics=['accuracy'])


# LSTM model
class RNN_LSTM(models.Model):
    def __init__(self):
        x = layers.Input(shape=(301, 62))
        h = layers.LSTM(8, dropout=0.2, recurrent_dropout=0.2, return_sequences=False)(x)
        h = layers.Dropout(0.5)(h)
        y = layers.Dense(2, activation='softmax')(h)
        super().__init__(x, y)

        rmsprop = optimizers.rmsprop(lr=0.0001)

        # try using different optimizers and different optimizer config
        self.compile(loss='binary_crossentropy',
                     optimizer='rmsprop', metrics=['accuracy'])




# load eeg files
def main():
    erp_raw, n_classes, n_sample, folds = Data()

    # for loop
    for j, (train_idx, test_idx) in enumerate(folds):
        print('\n Sampling Process ', j+1)
        train_subj = erp_raw[train_idx]
        test_subj = erp_raw[test_idx]

        onehot_encoder = OneHotEncoder()

        # erp sorting
        train_x = np.append(train_subj[:, :, :, 0], train_subj[:, :, :, 1], axis=0)
        train_y = np.append(np.zeros(train_subj.shape[0])+1, np.zeros(train_subj.shape[0])+2, axis=0)
        train_y = train_y.reshape(len(train_y), 1)
        train_y = onehot_encoder.fit_transform(train_y)
        test_x = np.append(test_subj[:, :, :, 0], test_subj[:, :, :, 1], axis=0)
        test_y = np.append(np.zeros(test_subj.shape[0])+1, np.zeros(test_subj.shape[0])+2, axis=0)
        test_y = test_y.reshape(len(test_y), 1)
        test_y = onehot_encoder.fit_transform(test_y)

        # erp shuffling
        train_x, train_y = shuffle(train_x, train_y)
        test_x, test_y = shuffle(test_x, test_y)

        # CNN modeling
        model = CNN_model()
        print('Training stage')
        print('==================')
        model.fit(train_x, train_y,
                  batch_size=4,
                  epochs=100,
                  shuffle=True,)

        score, acc = model.evaluate(test_x, test_y, batch_size=4)
        print('Test performance: accuracy={0}, loss={1}'.format(acc, score))

main()