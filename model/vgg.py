from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras import layers
from keras import models
from keras import Sequential
from keras import layers
from keras import optimizers
from keras import callbacks

def VGG16(lr, include_top=True,
          input_tensor=None, input_shape= (75, 75, 2),
          pooling=None,
          classes=1):
    """Instantiates the VGG16 architecture.
    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 input channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    input1 = layers.Input(shape=(75, 75, 2), name='data1')
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input1)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into ount
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = input1
    # Create model.
    model = Model(inputs=[input1], outputs=[x], name='vgg16')
    optimizer = optimizers.Adam(lr = lr)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])

    return model

def VGG9(dropout = 0.1, lr = 2e-5):
    input1 = layers.Input(shape=(75, 75, 2), name='Data1')

    db1 = layers.BatchNormalization(momentum=0.0)(input1)
    db1 = layers.Conv2D(32, (7,7), activation='relu', padding='same')(db1)
    db1 = layers.MaxPooling2D((2, 2))(db1)
    db1 = layers.Dropout(dropout)(db1)
    
    #db2 = layers.BatchNormalization(momentum=0.0)(db1)
    db2 = layers.Conv2D(64, (5,5), activation='relu', padding='same')(db1)
    db2 = layers.MaxPooling2D((2, 2))(db2)
    db2 = layers.Dropout(dropout)(db2)
    
    #db3 = layers.BatchNormalization(momentum=0.0)(db2)
    db3 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(db2)
    db3 = layers.MaxPooling2D((2, 2))(db3)
    db3 = layers.Dropout(dropout)(db3)
    db3 = layers.Flatten()(db3)

    fb1 = layers.Dense(128, activation='relu')(db3)
    fb1 = layers.Dropout(0.5)(fb1)
    output = layers.Dense(1, activation='sigmoid')(fb1)
    
    model = models.Model(inputs=[input1], outputs=[output])
    optimizer = optimizers.Adam(lr = lr)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
    return model