from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.layers import Reshape, Activation, Dropout, BatchNormalization
from keras.optimizers import *
from keras import regularizers
import keras

def get_MLP(input_shape=(150528,)):
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=input_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model

def model_VGG16(dropout_rate, l2_rate, BN_setting='NO_BN', input_tensor=Input(shape=(224,224,3)), activation='relu'):
    # get VGG16 model architecture

    inputs = input_tensor
    if l2_rate == 0.0:
        conv1 = Conv2D(64, (3, 3), padding='same', data_format='channels_last')(inputs)
    if l2_rate != 0.0:
        conv1 = Conv2D(64, (3, 3), padding='same', data_format='channels_last', kernel_regularizer=regularizers.l2(l2_rate))(inputs)

    if BN_setting == 'BN_ACT':
        conv1 = BatchNormalization()(conv1)
    conv1 = Activation(activation)(conv1)
    if BN_setting == 'ACT_BN':
        conv1 = BatchNormalization()(conv1)
    if l2_rate == 0.0:
        conv2 = Conv2D(64, (3, 3), padding='same', data_format='channels_last')(conv1)
    if l2_rate != 0.0:
        conv2 = Conv2D(64, (3, 3), padding='same', data_format='channels_last', kernel_regularizer=regularizers.l2(l2_rate))(conv1)
    if BN_setting == 'BN_ACT':
        conv2 = BatchNormalization()(conv2)
    conv2 = Activation(activation)(conv2)
    if BN_setting == 'ACT_BN':
        conv2 = BatchNormalization()(conv2)
    conv2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)

    if l2_rate == 0.0:
        conv3 = Conv2D(128, (3, 3), padding='same', data_format='channels_last')(conv2)
    if l2_rate != 0.0:
        conv3 = Conv2D(128, (3, 3), padding='same', data_format='channels_last', kernel_regularizer=regularizers.l2(l2_rate))(conv2)
    if BN_setting == 'BN_ACT':
        conv3 = BatchNormalization()(conv3)
    conv3 = Activation(activation)(conv3)
    if BN_setting == 'ACT_BN':
            conv3 = BatchNormalization()(conv3)
    if l2_rate == 0.0:
        conv4 = Conv2D(128, (3, 3), padding='same', data_format='channels_last')(conv3)
    if l2_rate != 0.0:
        conv4 = Conv2D(128, (3, 3), padding='same', data_format='channels_last', kernel_regularizer=regularizers.l2(l2_rate))(conv3)
    if BN_setting == 'BN_ACT':
        conv4 = BatchNormalization()(conv4)
    conv4 = Activation(activation)(conv4)
    if BN_setting == 'ACT_BN':
        conv4 = BatchNormalization()(conv4)
    conv4 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv4)

    if l2_rate == 0.0:
        conv5 = Conv2D(256, (3, 3), padding='same', data_format='channels_last')(conv4)
    if l2_rate != 0.0:
        conv5 = Conv2D(256, (3, 3), padding='same', data_format='channels_last', kernel_regularizer=regularizers.l2(l2_rate))(conv4)
    if BN_setting == 'BN_ACT':
        conv5 = BatchNormalization()(conv5)
    conv5 = Activation(activation)(conv5)
    if BN_setting == 'ACT_BN':
        conv5 = BatchNormalization()(conv5)
    if l2_rate == 0.0:
        conv6 = Conv2D(256, (3, 3), padding='same', data_format='channels_last')(conv5)
    if l2_rate != 0.0:
        conv6 = Conv2D(256, (3, 3), padding='same', data_format='channels_last', kernel_regularizer=regularizers.l2(l2_rate))(conv5)
    if BN_setting == 'BN_ACT':
        conv6 = BatchNormalization()(conv6)
    conv6 = Activation(activation)(conv6)
    if BN_setting == 'ACT_BN':
        conv6 = BatchNormalization()(conv6)
    if l2_rate == 0.0:
        conv7 = Conv2D(256, (3, 3), padding='same', data_format='channels_last')(conv6)
    if l2_rate != 0.0:
        conv7 = Conv2D(256, (3, 3), padding='same', data_format='channels_last', kernel_regularizer=regularizers.l2(l2_rate))(conv6)
    if BN_setting == 'BN_ACT':
        conv7 = BatchNormalization()(conv7)
    conv7 = Activation(activation)(conv7)
    if BN_setting == 'ACT_BN':
        conv7 = BatchNormalization()(conv7)
    conv7 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv7)

    if l2_rate == 0.0:
        conv8 = Conv2D(512, (3, 3), padding='same', data_format='channels_last')(conv7)
    if l2_rate != 0.0:
        conv8 = Conv2D(512, (3, 3), padding='same', data_format='channels_last', kernel_regularizer=regularizers.l2(l2_rate))(conv7)
    if BN_setting == 'BN_ACT':
        conv8 = BatchNormalization()(conv8)
    conv8 = Activation(activation)(conv8)
    if BN_setting == 'ACT_BN':
        conv8 = BatchNormalization()(conv8)
    if l2_rate == 0.0:
        conv9 = Conv2D(512, (3, 3), padding='same', data_format='channels_last')(conv8)
    if l2_rate != 0.0:
        conv9 = Conv2D(512, (3, 3), padding='same', data_format='channels_last', kernel_regularizer=regularizers.l2(l2_rate))(conv8)
    if BN_setting == 'BN_ACT':
        conv9 = BatchNormalization()(conv9)
    conv9 = Activation(activation)(conv9)
    if BN_setting == 'ACT_BN':
        conv9 = BatchNormalization()(conv9)
    if l2_rate == 0.0:
        conv10 = Conv2D(512, (3, 3), padding='same', data_format='channels_last')(conv9)
    if l2_rate != 0.0:
        conv10 = Conv2D(512, (3, 3), padding='same', data_format='channels_last', kernel_regularizer=regularizers.l2(l2_rate))(conv9)
    if BN_setting == 'BN_ACT':
        conv10 = BatchNormalization()(conv10)
    conv10 = Activation(activation)(conv10)
    if BN_setting == 'ACT_BN':
        conv10 = BatchNormalization()(conv10)
    conv10 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv10)

    if l2_rate == 0.0:
        conv11 = Conv2D(512, (3, 3), padding='same', data_format='channels_last')(conv10)
    if l2_rate != 0.0:
        conv11 = Conv2D(512, (3, 3), padding='same', data_format='channels_last', kernel_regularizer=regularizers.l2(l2_rate))(conv10)
    if BN_setting == 'BN_ACT':
        conv11 = BatchNormalization()(conv11)
    conv11 = Activation(activation)(conv11)
    if BN_setting == 'ACT_BN':
        conv11 = BatchNormalization()(conv11)
    if l2_rate == 0.0:
        conv12 = Conv2D(512, (3, 3), padding='same', data_format='channels_last')(conv11)
    if l2_rate != 0.0:
        conv12 = Conv2D(512, (3, 3), padding='same', data_format='channels_last', kernel_regularizer=regularizers.l2(l2_rate))(conv11)
    if BN_setting == 'BN_ACT':
        conv12 = BatchNormalization()(conv12)
    conv12 = Activation(activation)(conv12)
    if BN_setting == 'ACT_BN':
        conv12 = BatchNormalization()(conv12)
    if l2_rate == 0.0:
        conv13 = Conv2D(512, (3, 3), padding='same', data_format='channels_last')(conv12)
    if l2_rate != 0.0:
        conv13 = Conv2D(512, (3, 3), padding='same', data_format='channels_last', kernel_regularizer=regularizers.l2(l2_rate))(conv12)
    if BN_setting == 'BN_ACT':
        conv13 = BatchNormalization()(conv13)
    conv13 = Activation(activation)(conv13)
    if BN_setting == 'ACT_BN':
        conv13 = BatchNormalization()(conv13)
    conv13 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv13)

    flatten = Flatten()(conv13)
    fc1 = Dense(4096, activation=activation)(flatten)
    fc1 = Dropout(dropout_rate)(fc1)
    fc2 = Dense(4096, activation=activation)(fc1)
    fc2 = Dropout(dropout_rate)(fc2)
    predictions = Dense(1, activation='sigmoid')(fc2)

    model_VGG16 = Model(inputs=inputs, outputs=predictions)

    return model_VGG16

def model_VGG16_light(input_tensor=Input(shape=(224,224,3)), activation='relu'):
    # get VGG16 model architecture

    inputs = input_tensor
    conv1 = Conv2D(64, (3, 3), padding='same', data_format='channels_last')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation(activation)(conv1)
    conv2 = Conv2D(64, (3, 3), padding='same', data_format='channels_last')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation(activation)(conv2)
    conv2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)

    conv3 = Conv2D(128, (3, 3), padding='same', data_format='channels_last')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation(activation)(conv3)
    conv4 = Conv2D(128, (3, 3), padding='same', data_format='channels_last')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation(activation)(conv4)
    conv4 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv4)

    conv5 = Conv2D(256, (3, 3), padding='same', data_format='channels_last')(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation(activation)(conv5)
    conv6 = Conv2D(256, (3, 3), padding='same', data_format='channels_last')(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation(activation)(conv6)
    conv7 = Conv2D(256, (3, 3), padding='same', data_format='channels_last')(conv6)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation(activation)(conv7)
    conv7 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv7)

    conv8 = Conv2D(512, (3, 3), padding='same', data_format='channels_last')(conv7)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation(activation)(conv8)
    conv9 = Conv2D(512, (3, 3), padding='same', data_format='channels_last')(conv8)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation(activation)(conv9)
    conv10 = Conv2D(512, (3, 3), padding='same', data_format='channels_last')(conv9)
    conv10 = BatchNormalization()(conv10)
    conv10 = Activation(activation)(conv10)
    conv10 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv10)

    conv11 = Conv2D(512, (3, 3), padding='same', data_format='channels_last')(conv10)
    conv11 = BatchNormalization()(conv11)
    conv11 = Activation(activation)(conv11)
    conv12 = Conv2D(512, (3, 3), padding='same', data_format='channels_last')(conv11)
    conv12 = BatchNormalization()(conv12)
    conv12 = Activation(activation)(conv12)
    conv13 = Conv2D(512, (3, 3), padding='same', data_format='channels_last')(conv12)
    conv13 = BatchNormalization()(conv13)
    conv13 = Activation(activation)(conv13)
    conv13 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv13)

    flatten = Flatten()(conv13)
    fc1 = Dense(4096, activation=activation)(flatten)
    fc2 = Dense(4096, activation=activation)(fc1)
    predictions = Dense(1, activation='sigmoid')(fc2)

    model_VGG16 = Model(inputs=inputs, outputs=predictions)

    return model_VGG16
