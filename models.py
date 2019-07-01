from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.layers import Reshape, Activation, Dropout, BatchNormalization
from keras.optimizers import *
import keras

def model_VGG16(input_tensor=Input(shape=(224,224,3)), activation='relu'):
    # get VGG16 model architecture

    inputs = input_tensor
    conv1 = Conv2D(64, (3, 3), padding='same', data_format='channels_last', activation=activation)(inputs)
    conv2 = Conv2D(64, (3, 3), padding='same', data_format='channels_last', activation=activation)(conv1)
    conv2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)

    conv3 = Conv2D(128, (3, 3), padding='same', data_format='channels_last', activation=activation)(conv2)
    conv4 = Conv2D(128, (3, 3), padding='same', data_format='channels_last', activation=activation)(conv3)
    conv4 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv4)

    conv5 = Conv2D(256, (3, 3), padding='same', data_format='channels_last', activation=activation)(conv4)
    conv6 = Conv2D(256, (3, 3), padding='same', data_format='channels_last', activation=activation)(conv5)
    conv7 = Conv2D(256, (3, 3), padding='same', data_format='channels_last', activation=activation)(conv6)
    conv7 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv7)

    conv8 = Conv2D(512, (3, 3), padding='same', data_format='channels_last', activation=activation)(conv7)
    conv9 = Conv2D(512, (3, 3), padding='same', data_format='channels_last', activation=activation)(conv8)
    conv10 = Conv2D(512, (3, 3), padding='same', data_format='channels_last', activation=activation)(conv9)
    conv10 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv10)

    conv11 = Conv2D(512, (3, 3), padding='same', data_format='channels_last', activation=activation)(conv10)
    conv12 = Conv2D(512, (3, 3), padding='same', data_format='channels_last', activation=activation)(conv11)
    conv13 = Conv2D(512, (3, 3), padding='same', data_format='channels_last', activation=activation)(conv12)
    conv13 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv13)

    flatten = Flatten()(conv13)
    fc1 = Dense(4096, activation=activation)(flatten)
    fc2 = Dense(4096, activation=activation)(fc1)
    predictions = Dense(1000, activation='softmax')(fc2)

    model_VGG16 = Model(inputs=inputs, outputs=predictions)

    return model_VGG16
