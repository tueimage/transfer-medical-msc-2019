from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.layers import Activation, Dropout, BatchNormalization
from keras.optimizers import *
from keras import regularizers


class VGG16:
    """Create VGG16 model."""

    def __init__(self, dropout_rate=0.3, l2_rate=0.0, batchnorm=True, activation='relu', input_shape=(224, 224, 3)):
        """Initialize class."""
        self.dropout_rate = dropout_rate
        self.l2_rate = l2_rate
        self.batchnorm = batchnorm
        self.activation = activation
        self.input_shape = input_shape

    def conv_bn_act(self, input_tensor, feature_channels=64):
        """Create a stack of 3 layers, convolution + batch norm + activation."""
        if self.l2_rate != 0.0:
            conv = Conv2D(feature_channels, (3, 3), padding='same', data_format='channels_last', kernel_regularizer=regularizers.l2(self.l2_rate))(input_tensor)
        else:
            conv = Conv2D(feature_channels, (3, 3), padding='same', data_format='channels_last')(input_tensor)

        if self.batchnorm:
            conv = BatchNormalization()(conv)

        output_tensor = Activation(self.activation)(conv)
        return output_tensor

    def get_model(self):
        """Create the model architecture."""
        input_tensor = Input(shape=self.input_shape)

        output_tensor = self.conv_bn_act(input_tensor, feature_channels=64)
        output_tensor = self.conv_bn_act(output_tensor, feature_channels=64)
        output_tensor = MaxPooling2D(pool_size=(2, 2), padding='same')(output_tensor)

        output_tensor = self.conv_bn_act(output_tensor, feature_channels=128)
        output_tensor = self.conv_bn_act(output_tensor, feature_channels=128)
        output_tensor = MaxPooling2D(pool_size=(2, 2), padding='same')(output_tensor)

        output_tensor = self.conv_bn_act(output_tensor, feature_channels=256)
        output_tensor = self.conv_bn_act(output_tensor, feature_channels=256)
        output_tensor = self.conv_bn_act(output_tensor, feature_channels=256)
        output_tensor = MaxPooling2D(pool_size=(2, 2), padding='same')(output_tensor)

        output_tensor = self.conv_bn_act(output_tensor, feature_channels=512)
        output_tensor = self.conv_bn_act(output_tensor, feature_channels=512)
        output_tensor = self.conv_bn_act(output_tensor, feature_channels=512)
        output_tensor = MaxPooling2D(pool_size=(2, 2), padding='same')(output_tensor)

        output_tensor = self.conv_bn_act(output_tensor, feature_channels=512)
        output_tensor = self.conv_bn_act(output_tensor, feature_channels=512)
        output_tensor = self.conv_bn_act(output_tensor, feature_channels=512)
        output_tensor = MaxPooling2D(pool_size=(2, 2), padding='same')(output_tensor)

        flatten = Flatten()(output_tensor)
        fc1 = Dense(4096, activation=self.activation)(flatten)
        fc1 = Dropout(self.dropout_rate)(fc1)
        fc2 = Dense(4096, activation=self.activation)(fc1)
        fc2 = Dropout(self.dropout_rate)(fc2)
        predictions = Dense(1, activation='sigmoid')(fc2)

        model = Model(inputs=input_tensor, outputs=predictions)
        return model


def get_MLP(input_shape=(150528,)):
    """Create MLP."""
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=input_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model


def get_shiftCNN(input_shape=(224, 224, 3)):
    """Create shift detection model."""
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model
