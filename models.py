from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.layers import Reshape, Activation, Dropout, BatchNormalization
from keras.optimizers import *
from keras import regularizers
import keras

def model_VGG16(dropout_rate, l2_rate, input_tensor=Input(shape=(224,224,3)), activation='relu'):
    # get VGG16 model architecture

    inputs = input_tensor
    conv1 = Conv2D(64, (3, 3), padding='same', data_format='channels_last', kernel_regularizer=regularizers.l2(l2_rate))(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation(activation)(conv1)
    conv2 = Conv2D(64, (3, 3), padding='same', data_format='channels_last', kernel_regularizer=regularizers.l2(l2_rate))(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation(activation)(conv2)
    conv2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)

    conv3 = Conv2D(128, (3, 3), padding='same', data_format='channels_last', kernel_regularizer=regularizers.l2(l2_rate))(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation(activation)(conv3)
    conv4 = Conv2D(128, (3, 3), padding='same', data_format='channels_last', kernel_regularizer=regularizers.l2(l2_rate))(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation(activation)(conv4)
    conv4 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv4)

    conv5 = Conv2D(256, (3, 3), padding='same', data_format='channels_last', kernel_regularizer=regularizers.l2(l2_rate))(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation(activation)(conv5)
    conv6 = Conv2D(256, (3, 3), padding='same', data_format='channels_last', kernel_regularizer=regularizers.l2(l2_rate))(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation(activation)(conv6)
    conv7 = Conv2D(256, (3, 3), padding='same', data_format='channels_last', kernel_regularizer=regularizers.l2(l2_rate))(conv6)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation(activation)(conv7)
    conv7 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv7)

    conv8 = Conv2D(512, (3, 3), padding='same', data_format='channels_last', kernel_regularizer=regularizers.l2(l2_rate))(conv7)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation(activation)(conv8)
    conv9 = Conv2D(512, (3, 3), padding='same', data_format='channels_last', kernel_regularizer=regularizers.l2(l2_rate))(conv8)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation(activation)(conv9)
    conv10 = Conv2D(512, (3, 3), padding='same', data_format='channels_last', kernel_regularizer=regularizers.l2(l2_rate))(conv9)
    conv10 = BatchNormalization()(conv10)
    conv10 = Activation(activation)(conv10)
    conv10 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv10)

    conv11 = Conv2D(512, (3, 3), padding='same', data_format='channels_last', kernel_regularizer=regularizers.l2(l2_rate))(conv10)
    conv11 = BatchNormalization()(conv11)
    conv11 = Activation(activation)(conv11)
    conv12 = Conv2D(512, (3, 3), padding='same', data_format='channels_last', kernel_regularizer=regularizers.l2(l2_rate))(conv11)
    conv12 = BatchNormalization()(conv12)
    conv12 = Activation(activation)(conv12)
    conv13 = Conv2D(512, (3, 3), padding='same', data_format='channels_last', kernel_regularizer=regularizers.l2(l2_rate))(conv12)
    conv13 = BatchNormalization()(conv13)
    conv13 = Activation(activation)(conv13)
    conv13 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv13)

    flatten = Flatten()(conv13)
    fc1 = Dense(4096, activation=activation)(flatten)
    fc1 = Dropout(dropout_rate)(fc1)
    fc2 = Dense(4096, activation=activation)(fc1)
    fc2 = Dropout(dropout_rate)(fc2)
    predictions = Dense(1, activation='sigmoid')(fc2)

    model_VGG16 = Model(inputs=inputs, outputs=predictions)

    return model_VGG16
