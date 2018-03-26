from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.preprocessing import image

from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

import numpy as np
SEED=np.random.randint(1337)
np.random.seed(SEED)
import tensorflow as tf
tf.set_random_seed(SEED)

from model.vgg import VGG16, VGG9
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from utils import load, merge_bands, get_callbacks, print_and_save_history, SamplePairing
from keras.models import load_model

def train_generator(X_train, Y_train, transform = None):
    """Generator that returns transformed batches
        args: X_train: input images, batched (batch_size, W, H, C)
              Y_train: input labels
              transform: transformation to apply from the below dictionary
    """

    transforms = {'translate discrete': iaa.Affine(translate_px={"x": iap.Choice([9, 0, -9]), "y": iap.Choice([9, 0, -9])}),
                  'translate random': iaa.Affine(translate_px={"x": (-12, 12), "y": (-12, 12)}), 
                  'rotate': iaa.Affine(rotate=(-45, 45)),
                  'vertical flip': iaa.Flipud(0.5),
                  'horizontal flip':iaa.Fliplr(0.5),
                  'scale': iaa.Affine(scale=(0.95, 1.05)),
                  'blur': iaa.GaussianBlur(sigma = 3.0),
                  'combined': iaa.Sequential([ 
                                        iaa.Affine(translate_px={"x": iap.Choice([9, 0, -9]), "y": iap.Choice([9, 0, -9])}),
                                        iaa.Affine(scale=(0.95, 1.05))
                                        ])}
    if transform:
        if transform in ['original']:
            """no further transformation"""
            pass
        else:
            seq = transforms[transform]
            X_train = seq.augment_images(X_train)
            
    iter_ = image.ImageDataGenerator() 
    batch = iter_.flow(X_train, Y_train, batch_size = 16, seed = 1337) 
    while True:
        yield batch.next()

def train_model(path, **args):
    train_path = path + "train.json"
    kfold_history = []
    train = load(train_path)
    train_bands = np.stack(train.bands).squeeze()
    labels = train.is_iceberg.values
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    model = VGG9(dropout = 0.1, lr = 2e-5)
    
    for fold_id, (train_idx, val_idx) in enumerate(kf.split(train_bands, labels)):
        """k - fold stratified cross-validation"""
        X_train, Y_train = train_bands[train_idx], labels[train_idx]
        X_val, Y_val = train_bands[val_idx], labels[val_idx]
        if samplepairing:
            samplepairing_freq = 2 # switch on SamplePairing once in a while
            samplepairing_duration = 1 # switch on SamplePairing for how many epochs?
            samplepairing_end = 40 # end samplepairing after how many epochs (needs to be changed when samplepairing_duration is not 1)?
            for i in range(100):
                if not i % samplepairing_freq: 
                    if i < samplepairing_end:
                        print('-------------------------------------------------')
                        print('SamplePairing switched on for {} epoch(s)'.format(samplepairing_duration))
                        X_train_aug, Y_train_aug = SamplePairing(X_train, Y_train, 800)
                        history = model.fit_generator(train_generator(X_train_aug, Y_train_aug, transform), steps_per_epoch = 512,  epochs = samplepairing_duration, callbacks=get_callbacks(filepath= transform + '_samplepairing_fold_{}'.format(fold_id), patience=5),verbose=1,validation_data=(X_val, Y_val))
                        model.save('my_model.h5')
                        model = load_model('my_model.h5')
                        print('-------------------------------------------------')
                        continue
                history = model.fit_generator(train_generator(X_train, Y_train, transform), steps_per_epoch = 512,  epochs = 1, callbacks=get_callbacks(filepath= transform + '_samplepairing_fold_{}'.format(fold_id), patience=5),verbose=1,validation_data=(X_val, Y_val))
    
                model.save('my_model.h5')
                model = load_model('my_model.h5')
    
            kfold_history.append(history)
        else: # no SamplePairing
            history = model.fit_generator(
                train_generator(X_train, Y_train, transform),
                steps_per_epoch = 512,
                epochs = 100,
                validation_data = (X_val, Y_val),
                callbacks = get_callbacks(filepath= transform + '_1_fold_{}'.format(fold_id), patience = 5),
                verbose = 1)
        kfold_history.append(history)

if __name__ == '__main__':
    
    path = "/home/frank/data/statoil/"
    transform = 'rotate'
    samplepairing = True
    
    history = train_model(path, samplepairing = samplepairing, transform = transform)