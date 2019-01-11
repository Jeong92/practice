# library import
import scipy.io as sio  # load .mat file
import numpy as np
from keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# load data for classification using LSTM
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


# LSTM model
class RNN_LSTM(models.Model):
    def __init__(self):
        x = layers.Input(shape=(301, 62))
        h = layers.LSTM(8, dropout=0.2, recurrent_dropout=0.2, return_sequences=False)(x)
        h = layers.Dropout(0.5)(h)
        y = layers.Dense(1, activation='sigmoid')(h)
        super().__init__(x, y)

        # rsmprop = optimizers.rmsprop(lr=0.01)

        # try using different optimizers and different optimizer config
        self.compile(loss='binary_crossentropy',
                     optimizer='sgd', metrics=['accuracy'])


# load eeg files
def main():
    erp_raw, n_classes, n_sample, folds = Data()

    # for loop
    for j, (train_idx, test_idx) in enumerate(folds):
        print('\n Sampling Process ', j+1)
        train_subj = erp_raw[train_idx]
        test_subj = erp_raw[test_idx]

        # erp sorting
        train_x = np.append(train_subj[:, :, :, 0], train_subj[:, :, :, 1], axis=0)
        train_y = np.append(np.zeros(train_subj.shape[0])+1, np.zeros(train_subj.shape[0])+2, axis=0)
        test_x = np.append(test_subj[:, :, :, 0], test_subj[:, :, :, 1], axis=0)
        test_y = np.append(np.zeros(test_subj.shape[0])+1, np.zeros(test_subj.shape[0])+2, axis=0)

        # erp shuffling
        train_x, train_y = shuffle(train_x, train_y)
        test_x, test_y = shuffle(test_x, test_y)

        # LSTM modeling
        model = RNN_LSTM()
        print('Training stage')
        print('==================')
        model.fit(train_x, train_y,
                  batch_size=16,
                  epochs=8,
                  shuffle=True,
                  validation_split=0.2)
        score, acc = model.evaluate(test_x, test_y, batch_size=4)
        print(model.predict(test_x))
        print('Test performance: accuracy={0}, loss={1}'.format(acc, score))

main()
#score, acc, train_y, test_y = main()
