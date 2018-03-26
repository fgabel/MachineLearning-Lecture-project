from tensorflow.python.lib.io import file_io
import pandas as pd
import numpy as np
from keras.callbacks import CSVLogger, EarlyStopping
from sklearn.utils import shuffle

def merge_bands(band1, band2):
    b1 = np.array(band1).astype(np.float32)
    b2 = np.array(band2).astype(np.float32)
    return [np.stack([b1, b2], -1).reshape(75, 75, 2)]

def print_and_save_history(h):
    history_val_loss = []
    history_val_acc =  []
    hloss = zip(h.history['loss'], h.history['val_loss'], h.history['acc'], h.history['val_acc'])
    for idx, loss in enumerate(hloss):
        print('Step {} loss: {}, val_loss: {}'.format(idx, loss[0], loss[1]))
    history_val_loss.append(h.history['val_loss'])
    history_val_acc.append(h.history['val_acc'])
    return history_val_loss, history_val_acc
    
#for i in history:
#    val_loss, val_acc = print_and_save_history(i)

def load(train_file):
    with file_io.FileIO(train_file, mode='r') as stream:
        df = pd.read_json(stream).set_index('id')
        if 'bands' not in df.columns:
            df['bands'] = df.apply(lambda row: merge_bands(row['band_1'], row['band_2']), axis=1)
            df = df.drop(['band_1', 'band_2'], axis=1)
        return df

def get_callbacks(filepath, patience=10):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    #csv_logger = CSVLogger('C:/Users/Frank/Desktop/kaggle/statoil/statoil_keras_files/' + filepath + 'log.csv', append=True, separator=';')
    return [es] # , csv_logger

def SamplePairing(X_train, Y_train, N):
    """Randomly add two samples and concatenate them to X_train, Y_train
        args: X_train: original dataset 
            N number of samples per class to add
        
    """
    high = X_train.shape[0]
    for i in range(N):
        """add N samples using SamplePairing"""
        random1, random2 = np.random.randint(low = 0, high = high, size=2)
        new_sample = np.expand_dims((X_train[random1, :, :, :] + X_train[random2, :, :, :])/2, axis = 0)
        X_train = np.concatenate((X_train, new_sample), axis = 0)
        Y_train = np.concatenate((Y_train, (Y_train[random1],)), axis = 0)

    X_train, Y_train = shuffle(X_train, Y_train, random_state = 0)
    # shuffle
    return X_train, Y_train
