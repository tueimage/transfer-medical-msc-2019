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

# configgg = tf.ConfigProto()
# configgg.gpu_options.per_process_gpu_memory_fraction = 0.4
# session = tf.Session(config=configgg)
# K.set_session(session)


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

# read parameters for wanted dataset from config file
with open('config.json') as json_file:
    json = json.load(json_file)
    config = json[args['dataset']]
    if args['mode'] == 'transfer' or args['mode'] == 'fine_tuning' or args['mode'] == 'SVM':
        config_source = json[args['source_dataset']]

# create results directory if it doesn't exist
resultspath = os.path.join(os.path.dirname(os.getcwd()), 'results')
if not os.path.exists(resultspath):
    os.makedirs(resultspath)

# assign batch size
batchsize = int(args['batchsize'])

# set a random seed
sd=28

# get timestamp for saving stuff
timestamp = datetime.datetime.now().strftime("%y%m%d_%Hh%M")

# get paths to training, validation and testing directories
trainingpath = config['trainingpath']
validationpath = config['validationpath']
testpath = config['testpath']

# get total number of images in each split, needed to train in batches
num_training = len(glob.glob(os.path.join(trainingpath, '**/*.jpg')))
num_validation = len(glob.glob(os.path.join(validationpath, '**/*.jpg')))
num_test = len(glob.glob(os.path.join(testpath, '**/*.jpg')))

# initialize image data generator objects
gen_obj_training = ImageDataGenerator(rescale=1./255, featurewise_center=True)
gen_obj_test = ImageDataGenerator(rescale=1./255, featurewise_center=True)

# function for randomized search for optimal hyperparameters
if args['mode'] == 'random_search':
    # we need to fit generators to training data
    # from this mean and std, featurewise_center is calculated in the generator
    x_train = load_training_data(trainingpath)
    gen_obj_training.fit(x_train, seed=sd)
    gen_obj_test.fit(x_train, seed=sd)


    # create save directory if it doesn't exist
    if not os.path.exists(config['model_savepath']):
        os.makedirs(config['model_savepath'])

    # get path to save csv file with results
    csvpath = os.path.join(config['model_savepath'], 'randomsearch.csv')

    # initialize dataframe to save results for different hyperparameters
    search_records = pd.DataFrame(columns=['epochs', 'train_time', 'AUC', 'skl_AUC', 'train_accuracy', 'val_accuracy', 'learning_rate', 'dropout_rate', 'l2_rate', 'batchsize'])

    # add the headers to
    search_records.to_csv(csvpath, index=False)

    # test multiple models
    for i in range(100):
        limit_memory()
        session = tf.Session()
        K.set_session(session)
        with session.as_default():
            with session.graph.as_default():
                # need a different random seed everytime for hyperparameters, otherwise will still get same parameters every iteration
                np.random.seed(random.randint(0, 100000))

                # get random params
                learning_rate = 10 ** np.random.uniform(-7,-1)
                dropout_rate = np.random.uniform(0,1)
                l2_rate = 10 ** np.random.uniform(-2,-.3)
                # batchsize = 2 ** np.random.randint(6)
                batchsize = 32

                # get get either [batch norm before relu, after relu, or not at all]
                # if there is batch norm, dropout will not be used
                BN_nr = np.random.randint(2)
                BN_setting = [True, False][BN_nr]

                # if BN_setting != 'NO_BN':
                #     dropout_rate = 0.0
                #     l2_rate = 0.0
                
                # now choose between an optimizer
                OPT_nr = np.random.randint(2)
                OPT_setting = ['sgd_opt', 'adam_opt'][OPT_nr]

                # l2_rate = 0.0
                # dropout_rate = i * 0.1
                # OPT_setting = 'adam_opt'
                # BN_setting = 'BN_ACT'
                # learning_rate = 2e-7
                # batchsize = 32

                # now set a constant random seed, so things that may be variable are the same for every trained model, e.g. weight initialization
                os.environ['PYTHONHASHSEED'] = str(sd)
                np.random.seed(sd)
                random.seed(sd)
                tf.set_random_seed(sd)

                # build and train model and add results to csv file
                train_model(config, learning_rate, dropout_rate, l2_rate, batchsize, BN_setting, OPT_setting, gen_obj_training, gen_obj_test, csvpath)
        limit_memory()

# train model from scratch
if args['mode'] == 'from_scratch':
    # we need to fit generators to training data
    # from this mean and std, featurewise_center is calculated in the generator
    x_train = load_training_data(trainingpath)
    gen_obj_training.fit(x_train, seed=sd)
    gen_obj_test.fit(x_train, seed=sd)

    # set random seed for result reproducability
    os.environ['PYTHONHASHSEED'] = str(sd)
    np.random.seed(sd)
    random.seed(sd)
    tf.set_random_seed(sd)

    # initialize the image generators that load batches of images
    gen_training = gen_obj_training.flow_from_directory(
        trainingpath,
        class_mode="binary",
        target_size=(224,224),
        color_mode="rgb",
        shuffle=True,
        batch_size=batchsize)

    gen_validation = gen_obj_test.flow_from_directory(
        validationpath,
        class_mode="binary",
        target_size=(224,224),
        color_mode="rgb",
        shuffle=False,
        batch_size=1)

    gen_test = gen_obj_test.flow_from_directory(
        testpath,
        class_mode="binary",
        target_size=(224,224),
        color_mode="rgb",
        shuffle=False,
        batch_size=1)

    # set input shape for VGG16 model
    input_shape = (224,224,3)

    # set params
    learning_rate = 2e-7
    nr_epochs = 100

    OPT_setting = 'adam_opt'

    # load VGG16 model architecture
    model_VGG16 = VGG16().get_model(dropout_rate = 0.3, l2_rate=0.0, batchnorm=True, activation='relu', input_shape=input_shape)

    # model_VGG16 = model_VGG16(dropout_rate, l2_rate, BN_setting, input_tensor=input_tensor, activation='relu')
    print(model_VGG16.summary())

    # set optimizer and compile model
    print("compiling model...")
    sgd = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    RMSprop = RMSprop(lr=1e-6)
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    if OPT_setting == 'sgd_opt':
        model_VGG16.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
    if OPT_setting == 'adam_opt':
        model_VGG16.compile(loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"])

    # #calculate relative class weights for the imbalanced training data
    # class_weights = {}
    # for i in range(len(config['classes'])):
    #     # get path to the class images and get number of samples for that class
    #     classpath = os.path.join(trainingpath, config['classes'][i])
    #     num_class = len(glob.glob(os.path.join(classpath, '*.jpg')))
    #
    #     # add number of samples to dictionary
    #     class_weights[i] = num_class
    #
    # # find the biggest class and use that number of samples to calculate class weights
    # maxval = max(class_weights.values())
    # class_weights = {label:maxval/val for (label,val) in class_weights.items()}



    # train the model
    print("training model...")
    hist = model_VGG16.fit_generator(
        gen_training,
        # steps_per_epoch = num_training // batchsize,
        validation_data = gen_validation,
        # validation_steps = num_validation // batchsize,
        # class_weight=class_weights,
        epochs=nr_epochs,
        verbose=1)

    history = hist.history

    # create save directory if it doesn't exist
    if not os.path.exists(config['model_savepath']):
        os.makedirs(config['model_savepath'])

    # save history
    pd.DataFrame(history).to_csv(os.path.join(config['model_savepath'], '{}_history.csv'.format(args['dataset'])))

    # save trained model
    print("saving model...")
    savepath = os.path.join(config['model_savepath'], "{}_model_VGG16.h5".format(args['dataset']))
    model_VGG16.save(savepath)

    # create plot directory if it doesn't exist and plot training progress
    print("saving plots...")
    if not os.path.exists(config['plot_path']):
        os.makedirs(config['plot_path'])
    plotpath = os.path.join(config['plot_path'], "{}_training.png".format(args['dataset']))
    plot_training(history, nr_epochs, plotpath)

    # check the model on the validation data and use this for tweaking (not on test data)
    # this is for checking the best training settings; afterwards we can test on test set
    print("evaluating model...")

    # re-initialize validation generator with batch size 1 so all val images are used
    gen_validation = gen_obj_test.flow_from_directory(
        validationpath,
        class_mode="binary",
        target_size=(224,224),
        color_mode="rgb",
        shuffle=False,
        batch_size=1)

    # make predictions
    preds = model_VGG16.predict_generator(gen_validation, verbose=1)

    # get true labels
    true_labels = gen_validation.classes

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

if args['mode'] == 'evaluate':
    # load model
    print("loading model...")
    model_VGG16 = load_model(os.path.join(config['model_savepath'], args['input']))

    # we need to fit generators to training data
    # from this mean and std, featurewise_center is calculated in the generator
    x_train = load_training_data(trainingpath)
    gen_obj_training.fit(x_train, seed=sd)
    gen_obj_test.fit(x_train, seed=sd)

    # if evaluating from saved model, timestamp is retrieved from saved model's
    # name so saved plots will have same timestamp in name as trained model
    # timestamp = args['input'][:12]
    timestamp = args['dataset']

    # read history
    history = pd.read_csv(os.path.join(config['model_savepath'], '{}_history.csv'.format(timestamp)))

    # find number of epochs the model has trained for
    epochs = len(history['loss'])

    # check the model on the validation data and use this for tweaking (not on test data)
    # this is for checking the best training settings; afterwards we can test on test set
    print("evaluating model...")

    # initialize validation generator
    gen_validation = gen_obj_test.flow_from_directory(
        validationpath,
        class_mode="binary",
        target_size=(224,224),
        color_mode="rgb",
        shuffle=False,
        batch_size=1)

    # make predictions
    preds = model_VGG16.predict_generator(gen_validation, verbose=1)

    # get true labels
    true_labels = gen_validation.classes

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

    # create plot directory if it doesn't exist and plot training progress
    print("saving plots...")
    if not os.path.exists(config['plot_path']):
        os.makedirs(config['plot_path'])
    plotpath = os.path.join(config['plot_path'], "{}_training.png".format(args['dataset']))
    plot_training(history, epochs, plotpath)

if args['mode'] == 'transfer' or args['mode'] == 'fine_tuning' or args['mode'] == 'SVM':
    # we need to fit generators to the training data of source dataset
    # from this mean and std, featurewise_center is calculated in the generator
    trainingpath_source = config_source['trainingpath']
    x_train = load_training_data(trainingpath_source)
    gen_obj_training.fit(x_train, seed=sd)
    gen_obj_test.fit(x_train, seed=sd)

    if args['mode'] == 'transfer' or args['mode'] == 'SVM':
        # initialize the image generators that load batches of images
        gen_training = gen_obj_training.flow_from_directory(
            trainingpath,
            class_mode="binary",
            target_size=(224,224),
            color_mode="rgb",
            shuffle=False,
            batch_size=batchsize)

    # if we do fine-tuning, we need to shuffle to data when training
    if args['mode'] == 'fine_tuning':
        # initialize the image generators that load batches of images
        gen_training = gen_obj_training.flow_from_directory(
            trainingpath,
            class_mode="binary",
            target_size=(224,224),
            color_mode="rgb",
            shuffle=True,
            batch_size=batchsize)

    gen_validation = gen_obj_test.flow_from_directory(
        validationpath,
        class_mode="binary",
        target_size=(224,224),
        color_mode="rgb",
        shuffle=False,
        batch_size=1)

    gen_test = gen_obj_test.flow_from_directory(
        testpath,
        class_mode="binary",
        target_size=(224,224),
        color_mode="rgb",
        shuffle=False,
        batch_size=1)

    # now load the pre-trained source network
    print("loading source network...")
    modelpath = os.path.join(config_source['model_savepath'], '{}_model_VGG16.h5'.format(args['source_dataset']))
    orig_model = load_model(modelpath)
    orig_model.summary()

    if args['mode'] == 'transfer':
        # use pre-trained model to classify other data
        preds = orig_model.predict_generator(gen_validation, verbose=1)

        # get true labels
        true_labels = gen_validation.classes

        # calculate AUC and sklearn AUC
        fpr, tpr, thresholds, AUC = AUC(preds, true_labels)
        skfpr, sktpr, skthresholds, skAUC = skAUC(preds, true_labels)

        # calculate accuracy score
        acc = accuracy(preds, true_labels)

        # get path to save csv file for results
        transfer_csvpath = os.path.join(resultspath, 'transfer_results.csv')
        print(transfer_csvpath)

        # create csv file if it doesn't exist yet with the correct headers
        if not os.path.exists(transfer_csvpath):
            # initialize dataframe to save results for different combinations of datasets and add to csv file
            transfer_results = pd.DataFrame(columns=['target_dataset', 'source_dataset', 'AUC', 'skAUC', 'ACC'])
            transfer_results.to_csv(transfer_csvpath, index=False)

        # add new results to dataframe
        row = pd.Series({'target_dataset': args['dataset'],
                        'source_dataset': args['source_dataset'],
                        'AUC': AUC,
                        'skAUC': skAUC,
                        'ACC': acc})

        # read existing dataframe, add new row and save again
        transfer_results = pd.read_csv(transfer_csvpath)
        transfer_results = transfer_results.append(row, ignore_index=True)
        transfer_results.to_csv(transfer_csvpath, index=False)

    if args['mode'] == 'SVM':
        # get bottleneck features as output from a specific layer
        bottleneck_model = Model(inputs=orig_model.input, outputs=orig_model.get_layer('flatten_1').output)

        # use pre-trained model to extract features
        train_features = bottleneck_model.predict_generator(gen_training, verbose=1)
        val_features = bottleneck_model.predict_generator(gen_validation, verbose=1)

        # get true labels
        true_labels = gen_training.classes

        # scale the data to zero mean, unit variance for PCA
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        val_features = scaler.transform(val_features)

        # fit PCA
        pca = PCA(.95)
        pca.fit(train_features)

        # apply PCA to features and validation data
        reduced_train_features = pca.transform(train_features)
        reduced_val_features = pca.transform(val_features)

        # fit SVM classifier
        clf = svm.LinearSVC(penalty='l2', loss='squared_hinge', C=1.0).fit(reduced_train_features, true_labels)
        # clf = LogisticRegression(solver="liblinear").fit(reduced_train_features, true_labels)

        # make predictions using the trained SVM
        # pred_labels = clf.predict(reduced_val_features)
        preds = clf.decision_function(reduced_val_features)

        # get true validation labels
        true_labels_val = gen_validation.classes

        # calculate AUC and sklearn AUC
        fpr, tpr, thresholds, AUC = AUC(preds, true_labels)
        skfpr, sktpr, skthresholds, skAUC = skAUC(preds, true_labels)

        # calculate accuracy score
        acc = accuracy(preds, true_labels)

        # get path to save csv file for results
        SVM_csvpath = os.path.join(resultspath, 'SVM_results.csv')
        print(SVM_csvpath)

        # create csv file if it doesn't exist yet with the correct headers
        if not os.path.exists(SVM_csvpath):
            # initialize dataframe to save results for different combinations of datasets and add to csv file
            SVM_results = pd.DataFrame(columns=['target_dataset', 'source_dataset', 'AUC', 'skAUC', 'ACC'])
            SVM_results.to_csv(SVM_csvpath, index=False)

        # add new results to dataframe
        row = pd.Series({'target_dataset': args['dataset'],
                        'source_dataset': args['source_dataset'],
                        'AUC': AUC,
                        'skAUC': skAUC,
                        'ACC': acc})

        # read existing dataframe, add new row and save again
        SVM_results = pd.read_csv(SVM_csvpath)
        SVM_results = SVM_results.append(row, ignore_index=True)
        SVM_results.to_csv(SVM_csvpath, index=False)

    if args['mode'] == 'fine_tuning':

        # get bottleneck features as output from a specific layer
        base_model = Model(inputs=orig_model.input, outputs=orig_model.get_layer('max_pooling2d_5').output)

        # freeze all layers in the base model to exclude them from training
        for layer in base_model.layers:
            layer.trainable = False
            # if isinstance(layer, keras.layers.normalization.BatchNormalization):
            #     layer._per_input_updates = {}

        K.set_learning_phase(1)

        # build classifier model to put on top of the base model
        top_model = Flatten()(base_model.output)
        top_model = Dense(128, activation="relu")(top_model)
        # top_model = Dropout(0.3)(top_model)
        top_model = Dense(8, activation="relu")(top_model)
        # top_model = Dropout(0.3)(top_model)
        top_model = Dense(1, activation="sigmoid")(top_model)

        # add the model on top of the base model
        model = Model(inputs=base_model.input, outputs=top_model)

        model.summary()

        # compile model
        print("compiling model...")
        # adam = Adam(lr=1e-4) # 1e-4 werkt goed, 1e-3 en hoger totaal niet
        sgd = SGD(lr=1e-3, nesterov=True)
        model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])

        # train model (only the top) for a few epochs so the new layers get
        # initialized with learned values instead of randomly
        print("training top model...")
        hist = model.fit_generator(
            gen_training,
            validation_data = gen_validation,
            epochs=25,
            verbose=1)

        # K.set_learning_phase(0)

        # reset the testing generator for network evaluation using the val data
        print("evaluating after fine-tuning top model...")
        gen_validation.reset()

        # make predictions
        preds = model.predict_generator(gen_validation, verbose=1)

        # get true validation labels
        true_labels = gen_validation.classes

        # calculate AUC and sklearn AUC
        fpr, tpr, thresholds, AUC = AUC(preds, true_labels)
        skfpr, sktpr, skthresholds, skAUC = skAUC(preds, true_labels)

        # calculate accuracy score
        acc = accuracy(preds, true_labels)

        # unfreeze some layers and train some more
        for layer in base_model.layers[-10:]:
            layer.trainable = True
            # if isinstance(layer, keras.layers.normalization.BatchNormalization):
            #     layer.trainable = False

        # print which layers are trainable now
        for layer in base_model.layers:
            print("{}: {}".format(layer, layer.trainable))

        # K.set_learning_phase(1)

        # reset image generators before training again
        gen_training.reset()
        gen_validation.reset()

        # recompile model
        print("compiling model...")
        # adam = Adam(lr=1e-4) # 1e-4 werkt goed, 1e-3 en hoger totaal niet
        sgd = SGD(lr=1e-4, nesterov=True)
        model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])

        # train model for some more epochs
        print("training top model and unfrozen layers...")
        hist = model.fit_generator(
            gen_training,
            validation_data = gen_validation,
            epochs=25,
            verbose=1)

        # K.set_learning_phase(0)

        # reset the testing generator for network evaluation using the test data
        print("evaluating after fine-tuning top model...")
        gen_validation.reset()

        # make predictions
        preds_ft = model.predict_generator(gen_validation, verbose=1)

        # get true validation labels
        true_labels_ft = gen_validation.classes

        # calculate AUC and sklearn AUC
        fpr_ft, tpr_ft, thresholds_ft, AUC_ft = AUC(preds_ft, true_labels_ft)
        skfpr_ft, sktpr_ft, skthresholds_ft, skAUC_ft = skAUC(preds_ft, true_labels_ft)

        # calculate accuracy score
        acc_ft = accuracy(preds_ft, true_labels_ft)

        # get path to save csv file for results
        FT_csvpath = os.path.join(resultspath, 'FT_results.csv')
        print(FT_csvpath)

        # create csv file if it doesn't exist yet with the correct headers
        if not os.path.exists(FT_csvpath):
            # initialize dataframe to save results for different combinations of datasets and add to csv file
            FT_results = pd.DataFrame(columns=['target_dataset', 'source_dataset', 'AUC', 'skAUC', 'ACC', 'AUC_ft', 'skAUC_ft', 'ACC_ft'])
            FT_results.to_csv(FT_csvpath, index=False)

        # add new results to dataframe
        row = pd.Series({'target_dataset': args['dataset'],
                        'source_dataset': args['source_dataset'],
                        'AUC': AUC,
                        'skl_AUC': skAUC,
                        'ACC': acc,
                        'AUC_ft': AUC_ft,
                        'skl_AUC_ft': skAUC_ft,
                        'ACC_ft': acc_ft})

        # read existing dataframe, add new row and save again
        FT_results = pd.read_csv(FT_csvpath)
        FT_results = FT_results.append(row, ignore_index=True)
        FT_results.to_csv(FT_csvpath, index=False)
