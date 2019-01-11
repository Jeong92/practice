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
    eeg_dir = 'C:\\Users\\user\\PycharmProjects\\DeepLearning\\eeg_classification\\raw_eeg\\'  # eeg data directory
    print('erp directory: \n', eeg_dir, end='\n')

    onehot_encoder = OneHotEncoder()

    # load erp file
    eeg_file = 'eeg_data.mat'
    eeg_data = sio.loadmat(eeg_dir + eeg_file)
    eeg_x = eeg_data['eeg_x']
    eeg_x = np.transpose(eeg_x, (2, 1, 0))
    eeg_y = np.transpose(eeg_data['eeg_y'], (0, 1))
    # one-hot coding (y label)
    eeg_y = onehot_encoder.fit_transform(eeg_y)
    n_sample = np.shape(eeg_x)[0]
    print('Finished extracting %s samples of EEG data\n' % (n_sample))

    # split train-test data for resampling (N=10)

    x_train, x_test, y_train, y_test = train_test_split(
        eeg_x, eeg_y, test_size=0.1, shuffle=True)

    return x_train, x_test, y_train, y_test


# CNN model
class CNN_model(models.Model):
    def __init__(self):
        x = layers.Input(shape=(601, 62))  # input layer(channel last)
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




# load eeg files
def main():
    x_train, x_test, y_train, y_test = Data()

    # erp shuffling
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)

    # LSTM modeling
    model = CNN_model()
    print('Training stage')
    print('==================')
    model.fit(x_train, y_train,
              batch_size=100,
              epochs=100,
              shuffle=True,
              validation_split=0.1)
    score, acc = model.evaluate(x_test, y_test, batch_size=4)
    print(model.predict(x_test))
    print(y_test)
    print('Test performance: accuracy={0}, loss={1}'.format(acc, score))

main()