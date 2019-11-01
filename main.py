from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model, Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers import Input
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import TensorBoard, EarlyStopping
from keras import backend as K
import keras
import tensorflow as tf
import json
import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import glob
from models import VGG16
import datetime
import random
import pandas as pd
import cv2
import gc
from time import time
from utils import *
from sklearn import svm

# choose GPU for training
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# construct argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m',
    '--mode',
    choices=['from_scratch', 'transfer', 'SVM', 'fine_tuning', 'evaluate', 'random_search'],
    required=True,
    help='training mode')
parser.add_argument('-d',
    '--dataset',
    choices=['isic',
            'isic_2',
            'ISIC_image_rot_f=0.1',
            'ISIC_image_rot_f=0.2',
            'ISIC_image_rot_f=0.3',
            'ISIC_image_rot_f=0.4',
            'ISIC_image_rot_f=0.5',
            'ISIC_image_rot_f=0.6',
            'ISIC_image_rot_f=0.7',
            'ISIC_image_rot_f=0.8',
            'ISIC_image_rot_f=0.9',
            'ISIC_image_rot_f=1.0',
            'ISIC_image_translation_f=0.1',
            'ISIC_image_translation_f=0.2',
            'ISIC_image_translation_f=0.3',
            'ISIC_image_translation_f=0.4',
            'ISIC_image_translation_f=0.5',
            'ISIC_image_translation_f=0.6',
            'ISIC_image_translation_f=0.7',
            'ISIC_image_translation_f=0.8',
            'ISIC_image_translation_f=0.9',
            'ISIC_image_translation_f=1.0',
            'ISIC_image_zoom_f=0.1',
            'ISIC_image_zoom_f=0.2',
            'ISIC_image_zoom_f=0.3',
            'ISIC_image_zoom_f=0.4',
            'ISIC_image_zoom_f=0.5',
            'ISIC_image_zoom_f=0.6',
            'ISIC_image_zoom_f=0.7',
            'ISIC_image_zoom_f=0.8',
            'ISIC_image_zoom_f=0.9',
            'ISIC_image_zoom_f=1.0',
            'ISIC_add_noise_gaussian_f=0.1',
            'ISIC_add_noise_gaussian_f=0.2',
            'ISIC_add_noise_gaussian_f=0.3',
            'ISIC_add_noise_gaussian_f=0.4',
            'ISIC_add_noise_gaussian_f=0.5',
            'ISIC_add_noise_gaussian_f=0.6',
            'ISIC_add_noise_gaussian_f=0.7',
            'ISIC_add_noise_gaussian_f=0.8',
            'ISIC_add_noise_gaussian_f=0.9',
            'ISIC_add_noise_gaussian_f=1.0',
            'ISIC_add_noise_poisson_f=0.1',
            'ISIC_add_noise_poisson_f=0.2',
            'ISIC_add_noise_poisson_f=0.3',
            'ISIC_add_noise_poisson_f=0.4',
            'ISIC_add_noise_poisson_f=0.5',
            'ISIC_add_noise_poisson_f=0.6',
            'ISIC_add_noise_poisson_f=0.7',
            'ISIC_add_noise_poisson_f=0.8',
            'ISIC_add_noise_poisson_f=0.9',
            'ISIC_add_noise_poisson_f=1.0',
            'ISIC_add_noise_salt_and_pepper_f=0.1',
            'ISIC_add_noise_salt_and_pepper_f=0.2',
            'ISIC_add_noise_salt_and_pepper_f=0.3',
            'ISIC_add_noise_salt_and_pepper_f=0.4',
            'ISIC_add_noise_salt_and_pepper_f=0.5',
            'ISIC_add_noise_salt_and_pepper_f=0.6',
            'ISIC_add_noise_salt_and_pepper_f=0.7',
            'ISIC_add_noise_salt_and_pepper_f=0.8',
            'ISIC_add_noise_salt_and_pepper_f=0.9',
            'ISIC_add_noise_salt_and_pepper_f=1.0',
            'ISIC_add_noise_speckle_f=0.1',
            'ISIC_add_noise_speckle_f=0.2',
            'ISIC_add_noise_speckle_f=0.3',
            'ISIC_add_noise_speckle_f=0.4',
            'ISIC_add_noise_speckle_f=0.5',
            'ISIC_add_noise_speckle_f=0.6',
            'ISIC_add_noise_speckle_f=0.7',
            'ISIC_add_noise_speckle_f=0.8',
            'ISIC_add_noise_speckle_f=0.9',
            'ISIC_add_noise_speckle_f=1.0',
            'ISIC_imbalance_classes_f=0.1',
            'ISIC_imbalance_classes_f=0.2',
            'ISIC_imbalance_classes_f=0.3',
            'ISIC_imbalance_classes_f=0.4',
            'ISIC_imbalance_classes_f=0.5',
            'ISIC_imbalance_classes_f=0.6',
            'ISIC_imbalance_classes_f=0.7',
            'ISIC_imbalance_classes_f=0.8',
            'ISIC_imbalance_classes_f=0.9',
            'ISIC_imbalance_classes_f=1.0',
            'isic_2017',
            'isic_2017_adj',
            'cats_and_dogs'],
    required=True,
    help='dataset to use, when using transfer, fine_tuning or SVM, this is the target dataset')
parser.add_argument('-s',
    '--source_dataset',
    choices=['isic',
            'ISIC_image_rot_f=0.1',
            'ISIC_image_rot_f=0.2',
            'ISIC_image_rot_f=0.3',
            'ISIC_image_rot_f=0.4',
            'ISIC_image_rot_f=0.5',
            'ISIC_image_rot_f=0.6',
            'ISIC_image_rot_f=0.7',
            'ISIC_image_rot_f=0.8',
            'ISIC_image_rot_f=0.9',
            'ISIC_image_rot_f=1.0',
            'ISIC_image_translation_f=0.1',
            'ISIC_image_translation_f=0.2',
            'ISIC_image_translation_f=0.3',
            'ISIC_image_translation_f=0.4',
            'ISIC_image_translation_f=0.5',
            'ISIC_image_translation_f=0.6',
            'ISIC_image_translation_f=0.7',
            'ISIC_image_translation_f=0.8',
            'ISIC_image_translation_f=0.9',
            'ISIC_image_translation_f=1.0',
            'ISIC_image_zoom_f=0.1',
            'ISIC_image_zoom_f=0.2',
            'ISIC_image_zoom_f=0.3',
            'ISIC_image_zoom_f=0.4',
            'ISIC_image_zoom_f=0.5',
            'ISIC_image_zoom_f=0.6',
            'ISIC_image_zoom_f=0.7',
            'ISIC_image_zoom_f=0.8',
            'ISIC_image_zoom_f=0.9',
            'ISIC_image_zoom_f=1.0',
            'ISIC_add_noise_gaussian_f=0.1',
            'ISIC_add_noise_gaussian_f=0.2',
            'ISIC_add_noise_gaussian_f=0.3',
            'ISIC_add_noise_gaussian_f=0.4',
            'ISIC_add_noise_gaussian_f=0.5',
            'ISIC_add_noise_gaussian_f=0.6',
            'ISIC_add_noise_gaussian_f=0.7',
            'ISIC_add_noise_gaussian_f=0.8',
            'ISIC_add_noise_gaussian_f=0.9',
            'ISIC_add_noise_gaussian_f=1.0',
            'ISIC_add_noise_poisson_f=0.1',
            'ISIC_add_noise_poisson_f=0.2',
            'ISIC_add_noise_poisson_f=0.3',
            'ISIC_add_noise_poisson_f=0.4',
            'ISIC_add_noise_poisson_f=0.5',
            'ISIC_add_noise_poisson_f=0.6',
            'ISIC_add_noise_poisson_f=0.7',
            'ISIC_add_noise_poisson_f=0.8',
            'ISIC_add_noise_poisson_f=0.9',
            'ISIC_add_noise_poisson_f=1.0',
            'ISIC_add_noise_salt_and_pepper_f=0.1',
            'ISIC_add_noise_salt_and_pepper_f=0.2',
            'ISIC_add_noise_salt_and_pepper_f=0.3',
            'ISIC_add_noise_salt_and_pepper_f=0.4',
            'ISIC_add_noise_salt_and_pepper_f=0.5',
            'ISIC_add_noise_salt_and_pepper_f=0.6',
            'ISIC_add_noise_salt_and_pepper_f=0.7',
            'ISIC_add_noise_salt_and_pepper_f=0.8',
            'ISIC_add_noise_salt_and_pepper_f=0.9',
            'ISIC_add_noise_salt_and_pepper_f=1.0',
            'ISIC_add_noise_speckle_f=0.1',
            'ISIC_add_noise_speckle_f=0.2',
            'ISIC_add_noise_speckle_f=0.3',
            'ISIC_add_noise_speckle_f=0.4',
            'ISIC_add_noise_speckle_f=0.5',
            'ISIC_add_noise_speckle_f=0.6',
            'ISIC_add_noise_speckle_f=0.7',
            'ISIC_add_noise_speckle_f=0.8',
            'ISIC_add_noise_speckle_f=0.9',
            'ISIC_add_noise_speckle_f=1.0',
            'ISIC_imbalance_classes_f=0.1',
            'ISIC_imbalance_classes_f=0.2',
            'ISIC_imbalance_classes_f=0.3',
            'ISIC_imbalance_classes_f=0.4',
            'ISIC_imbalance_classes_f=0.5',
            'ISIC_imbalance_classes_f=0.6',
            'ISIC_imbalance_classes_f=0.7',
            'ISIC_imbalance_classes_f=0.8',
            'ISIC_imbalance_classes_f=0.9',
            'ISIC_imbalance_classes_f=1.0',
            'isic_2017',
            'isic_2017_adj',
            'cats_and_dogs'],
    required='transfer' in sys.argv or 'fine_tuning' in sys.argv or 'SVM' in sys.argv,
    help='source dataset to use when using transfer, fine_tuning or SVM')
parser.add_argument('-i',
    '--input',
    required='evaluate' in sys.argv,
    help='name of trained model to load when evaluating')
parser.add_argument('-bs', '--batchsize', default=32, help='batch size')
args = vars(parser.parse_args())

class NeuralNetwork:
    def __init__(self, model, trainingpath, validationpath, **kwargs):
        self.model = model
        self.trainingpath = trainingpath
        self.validationpath = validationpath
        for attribute, value in kwargs.items():
            setattr(self, attribute, value)

        # initialize image data generator objects
        gen_obj_training = ImageDataGenerator(rescale=1./255, featurewise_center=True)
        gen_obj_test = ImageDataGenerator(rescale=1./255, featurewise_center=True)

        # fit generators to training data
        x_train = load_training_data(self.trainingpath)
        gen_obj_training.fit(x_train, seed=self.seed)
        gen_obj_test.fit(x_train, seed=self.seed)

        # initialize image generators that load batches of images
        self.gen_training = gen_obj_training.flow_from_directory(
            self.trainingpath,
            class_mode="binary",
            target_size=(224,224),
            color_mode="rgb",
            shuffle=True,
            batch_size=self.batchsize)

        self.gen_validation = gen_obj_test.flow_from_directory(
            self.validationpath,
            class_mode="binary",
            target_size=(224,224),
            color_mode="rgb",
            shuffle=False,
            batch_size=1)

    def compile_network(self, learning_rate, **kwargs):
        # compile model
        self.optimizer = kwargs.get('optimizer', 'adam')
        self.loss = kwargs.get('loss', "binary_crossentropy")
        self.metrics = kwargs.get('metrics', ["accuracy"])
        self.learning_rate = learning_rate

        # check which optimizer should be used
        if self.optimizer == 'adam':
            self.optimizer = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        elif self.optimizer == 'rmsprop':
            self.optimizer = RMSprop(lr=self.learning_rate)
        elif self.optimizer == 'sgd':
            self.optimizer = SGD(lr=self.learning_rate, momentum=0.9, nesterov=True)
        else:
            print("Unsupported optimizer, please choose one of (adam, rmsprop, sgd)")

        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

    def train(self, epochs):
        # get total number of images in each split, needed to train in batches
        num_training = len(glob.glob(os.path.join(self.trainingpath, '**/*.jpg')))
        num_validation = len(glob.glob(os.path.join(self.validationpath, '**/*.jpg')))

        # train the model
        hist = self.model.fit_generator(self.gen_training,
            steps_per_epoch = num_training // self.batchsize,
            validation_data=self.gen_validation,
            validation_steps = num_validation // self.batchsize,
            epochs=epochs,
            verbose=1)

        history = hist.history

        # save history
        pd.DataFrame(history).to_csv(os.path.join(config['model_savepath'], '{}_history.csv'.format(args['dataset'])))

        return history

    def save_model(self):
        # save trained model
        print("saving model...")
        savepath = os.path.join(config['model_savepath'], "{}_model.h5".format(args['dataset']))
        model.save(savepath)

    def plot_training(self, history):
        # plot and save training history
        plt.style.use("ggplot")

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

        ax1.set_ylabel('Loss')
        ax1.set_xlim([1,epochs])
        ax1.plot(np.arange(0, epochs), history["loss"], label="train")
        ax1.plot(np.arange(0, epochs), history["val_loss"], label="validation")
        ax1.legend(loc="upper right")

        ax2.set_ylabel('Accuracy')
        ax2.set_ylim([0, 1])
        ax2.plot(np.arange(0, epochs), history["acc"], label="train")
        ax2.plot(np.arange(0, epochs), history["val_acc"], label="validation")
        ax2.legend(loc="lower right")

        # create plot path
        plotpath = os.path.join(config['plot_path'], "{}_training.png".format(args['dataset']))

        # save plot
        plt.savefig(plotpath)

    def evaluate(self):
        # make predictions
        preds = model.predict_generator(self.gen_validation, verbose=1)

        # get true labels
        true_labels = self.gen_validation.classes

        # calculate AUC and sklearn AUC
        fpr, tpr, thresholds, AUC = AUC(preds, true_labels)
        skfpr, sktpr, skthresholds, skAUC = skAUC(preds, true_labels)

        # calculate accuracy score
        acc = accuracy(preds, true_labels)

        # plot AUC plots
        savepath = os.path.join(config['plot_path'], "{}_ROC.png".format(args['dataset']))
        plot_AUC(fpr, tpr, AUC, savepath)

        sksavepath = os.path.join(config['plot_path'], "{}_skROC.png".format(args['dataset']))
        plot_skAUC(skfpr, sktpr, skAUC, sksavepath)

def main():
    # read parameters for wanted dataset from config file
    with open('config.json') as json_file:
        config_json = json.load(json_file)
        config = config_json[args['dataset']]
        if args['mode'] in ['SVM', 'fine_tuning']:
            config_source = config_json[args['source_dataset']]

    # create directories to save results if they don't exist yet
    resultspath = os.path.join(os.path.dirname(os.getcwd()), 'results')
    if not os.path.exists(resultspath):
        os.makedirs(resultspath)

    if not os.path.exists(config['model_savepath']):
        os.makedirs(config['model_savepath'])

    if not os.path.exists(config['plot_path']):
        os.makedirs(config['plot_path'])

    # assign batch size
    batchsize = int(args['batchsize'])

    # set a random seed
    seed=28

    # get paths to training, validation and testing directories
    trainingpath = config['trainingpath']
    validationpath = config['validationpath']
    testpath = config['testpath']


    if args['mode'] == 'from_scratch':
        # set random seed for result reproducability
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        random.seed(seed)
        tf.set_random_seed(seed)

        # set parameters for training
        learning_rate = 2e-7
        epochs = 100

        # load VGG16 model architecture
        model = VGG16(dropout_rate=0.3, l2_rate=0.0, batchnorm=True, activation='relu', input_shape=(224,224,3)).get_model()
        model.summary()

        # create network instance
        network = NeuralNetwork(model, trainingpath, validationpath, batchsize=batchsize, seed=seed)

        # compile network
        print("compiling network...")
        network.compile_network(learning_rate, optimizer='adam', loss="binary_crossentropy", metrics=["accuracy"])

        # train network
        print("training network...")
        history = network.train(epochs)

        # save network
        print("saving network...")
        network.save_model()

        # plot training
        print("plotting training progress...")
        network.plot_training(history)

        # evaluate network on validation data
        print("evaluating network on validation data...")
        network.evaluate()

if __name__ == "__main__":
    main()
