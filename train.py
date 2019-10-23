from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.applications import VGG16
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
from models import model_VGG16
import datetime
import random
import pandas as pd
import cv2
import gc
from time import time
from helper_functions import ROC_AUC, load_data, load_training_data, plot_training, train_model, limit_memory
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
    for i in range(10):
        limit_memory()
        session = tf.Session()
        K.set_session(session)
        with session.as_default():
            with session.graph.as_default():
                # need a different random seed everytime for hyperparameters, otherwise will still get same parameters every iteration
                np.random.seed(random.randint(0, 100000))

                # # get random params
                # learning_rate = 10 ** np.random.uniform(-7,-1)
                # dropout_rate = np.random.uniform(0,1)
                # l2_rate = 10 ** np.random.uniform(-2,-.3)
                # # batchsize = 2 ** np.random.randint(6)
                # batchsize = 32
                #
                # # get get either [batch norm before relu, after relu, or not at all]
                # # if there is batch norm, dropout will not be used
                # BN_nr = np.random.randint(3)
                # BN_setting = ['BN_ACT', 'ACT_BN', 'NO_BN'][BN_nr]
                #
                # if BN_setting != 'NO_BN':
                #     dropout_rate = 0.0
                #     l2_rate = 0.0
                #
                # # now choose between an optimizer
                # OPT_nr = np.random.randint(2)
                # OPT_setting = ['sgd_opt', 'adam_opt'][OPT_nr]

                l2_rate = 0.0
                dropout_rate = i * 0.1
                OPT_setting = 'adam_opt'
                BN_setting = 'BN_ACT'
                learning_rate = 2e-7
                batchsize = 32

                # # now set a constant random seed, so things that may be variable are the same for every trained model, e.g. weight initialization
                # os.environ['PYTHONHASHSEED'] = str(sd)
                # np.random.seed(sd)
                # random.seed(sd)
                # tf.set_random_seed(sd)

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

    # get timestamp for saving stuff
    timestamp = args['dataset']

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

    # set input tensor for VGG16 model
    input_tensor = Input(shape=(224,224,3))

    # set params
    dropout_rate = 0.3
    l2_rate = 0.0
    learning_rate = 2e-7
    nr_epochs = 100

    BN_setting = 'BN_ACT'
    OPT_setting = 'adam_opt'

    # load VGG16 model architecture
    model_VGG16 = model_VGG16(dropout_rate, l2_rate, BN_setting, input_tensor=input_tensor, activation='relu')
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

    # tensorboard = TensorBoard(log_dir="logs/{}".format(time()), batch_size=batchsize, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

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

    # plot ROC and calculate AUC
    ROC_AUC(preds, true_labels, config, timestamp)

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

    # plot ROC and calculate AUC
    ROC_AUC(preds, true_labels, config, timestamp)

    # create plot directory if it doesn't exist and plot training progress
    print("saving plots...")
    if not os.path.exists(config['plot_path']):
        os.makedirs(config['plot_path'])
    plotpath = os.path.join(config['plot_path'], "{}_training.png".format(timestamp))
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

        # plot ROC and calculate AUC
        AUC, AUC2 = ROC_AUC(preds, true_labels, config, timestamp)

        # preds is an array like [[x] [x] [x]], make it into array like [x x x]
        preds = np.asarray([label for sublist in preds for label in sublist])

        # calculate accuracy
        pred_labels = np.where(preds > 0.5, 1, 0).astype(int)

        # calculate True Positive (TP), True Negative (TN), False Positive (FP) and
        # False Negative (FN)
        TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
        TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
        FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
        FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))

        ACC = round(((TP + TN) / (TP + TN + FP + FN)),3)

        # get path to save csv file for results
        transfer_csvpath = os.path.join(resultspath, 'transfer_results.csv')
        print(transfer_csvpath)

        # create csv file if it doesn't exist yet with the correct headers
        if not os.path.exists(transfer_csvpath):
            # initialize dataframe to save results for different combinations of datasets and add to csv file
            transfer_results = pd.DataFrame(columns=['target_dataset', 'source_dataset', 'AUC', 'skl_AUC', 'ACC'])
            transfer_results.to_csv(transfer_csvpath, index=False)

        # add new results to dataframe
        row = pd.Series({'target_dataset': args['dataset'],
                        'source_dataset': args['source_dataset'],
                        'AUC': AUC,
                        'skl_AUC': AUC2,
                        'ACC': ACC})

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

        # initialize TPR, FPR, ACC and AUC lists
        TPR_list, FPR_list, ACC_list = [], [], []
        AUC_score = []

        # calculate for different thresholds
        thresholds = -np.sort(-(np.unique(preds)))
        for threshold in thresholds:
            # apply threshold to predictions
            pred_labels = np.where(preds > threshold, 1, 0).astype(int)

            # calculate True Positive (TP), True Negative (TN), False Positive (FP) and
            # False Negative (FN)
            TP = np.sum(np.logical_and(pred_labels == 1, true_labels_val == 1))
            TN = np.sum(np.logical_and(pred_labels == 0, true_labels_val == 0))
            FP = np.sum(np.logical_and(pred_labels == 1, true_labels_val == 0))
            FN = np.sum(np.logical_and(pred_labels == 0, true_labels_val == 1))

            # calculate TPR, FPR, ACC and add to lists
            TPR = TP / (TP + FN)
            FPR = FP / (FP + TN)
            ACC = (TP + TN) / (TP + TN + FP + FN)

            TPR_list.append(TPR)
            FPR_list.append(FPR)
            ACC_list.append(ACC)

            AUC_score.append((1-FPR+TPR)/2)

            pred_labels = []

        AUC = round(sum(AUC_score)/len(thresholds),3)
        print("AUC: {}".format(AUC))

        pred_labels = np.where(preds > 0.5, 1, 0).astype(int)

        # now with sklearn implementation
        fpr, tpr, thresholds = roc_curve(true_labels_val, preds, pos_label=1)

        AUC2 = round(roc_auc_score(true_labels_val, preds),3)
        print("sk_AUC: {}".format(AUC2))

        # calculate True Positive (TP), True Negative (TN), False Positive (FP) and
        # False Negative (FN)
        TP = np.sum(np.logical_and(pred_labels == 1, true_labels_val == 1))
        TN = np.sum(np.logical_and(pred_labels == 0, true_labels_val == 0))
        FP = np.sum(np.logical_and(pred_labels == 1, true_labels_val == 0))
        FN = np.sum(np.logical_and(pred_labels == 0, true_labels_val == 1))

        # TPR = TP / (TP + FN)
        # FPR = FP / (FP + TN)
        #
        # AUC = round(((1-FPR+TPR)/2),3)
        #
        # print(AUC)

        ACC = round(((TP + TN) / (TP + TN + FP + FN)),3)

        print(ACC)


        # get path to save csv file for results
        SVM_csvpath = os.path.join(resultspath, 'SVM_results.csv')
        print(SVM_csvpath)

        # create csv file if it doesn't exist yet with the correct headers
        if not os.path.exists(SVM_csvpath):
            # initialize dataframe to save results for different combinations of datasets and add to csv file
            SVM_results = pd.DataFrame(columns=['target_dataset', 'source_dataset', 'AUC', 'skl_AUC', 'ACC'])
            SVM_results.to_csv(SVM_csvpath, index=False)

        # add new results to dataframe
        row = pd.Series({'target_dataset': args['dataset'],
                        'source_dataset': args['source_dataset'],
                        'AUC': AUC,
                        'skl_AUC': AUC2,
                        'ACC': ACC})

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

        # preds is an array like [[x] [x] [x]], make it into array like [x x x]
        preds = np.asarray([label for sublist in preds for label in sublist])

        # get true validation labels
        true_labels_val = gen_validation.classes

        # initialize TPR, FPR, ACC and AUC lists
        TPR_list, FPR_list, ACC_list = [], [], []
        AUC_score = []

        # calculate for different thresholds
        thresholds = -np.sort(-(np.unique(preds)))
        for threshold in thresholds:
            # apply threshold to predictions
            pred_labels = np.where(preds > threshold, 1, 0).astype(int)

            # calculate True Positive (TP), True Negative (TN), False Positive (FP) and
            # False Negative (FN)
            TP = np.sum(np.logical_and(pred_labels == 1, true_labels_val == 1))
            TN = np.sum(np.logical_and(pred_labels == 0, true_labels_val == 0))
            FP = np.sum(np.logical_and(pred_labels == 1, true_labels_val == 0))
            FN = np.sum(np.logical_and(pred_labels == 0, true_labels_val == 1))

            # calculate TPR, FPR, ACC and add to lists
            TPR = TP / (TP + FN)
            FPR = FP / (FP + TN)
            ACC = (TP + TN) / (TP + TN + FP + FN)

            TPR_list.append(TPR)
            FPR_list.append(FPR)
            ACC_list.append(ACC)

            AUC_score.append((1-FPR+TPR)/2)

            pred_labels = []

        AUC = round(sum(AUC_score)/len(thresholds),3)
        print("AUC: {}".format(AUC))

        pred_labels = np.where(preds > 0.5, 1, 0).astype(int)

        # now with sklearn implementation
        fpr, tpr, thresholds = roc_curve(true_labels_val, preds, pos_label=1)

        AUC2 = round(roc_auc_score(true_labels_val, preds),3)
        print("sk_AUC: {}".format(AUC2))

        # calculate True Positive (TP), True Negative (TN), False Positive (FP) and
        # False Negative (FN)
        TP = np.sum(np.logical_and(pred_labels == 1, true_labels_val == 1))
        TN = np.sum(np.logical_and(pred_labels == 0, true_labels_val == 0))
        FP = np.sum(np.logical_and(pred_labels == 1, true_labels_val == 0))
        FN = np.sum(np.logical_and(pred_labels == 0, true_labels_val == 1))

        ACC1 = round(((TP + TN) / (TP + TN + FP + FN)),3)

        print(ACC1)


        # for i in range(len(model.layers)):
        #     layer = model.layers[i]
        #     # skip non-conv layers
        #     if 'conv' not in layer.name:
        #         continue
        #     print(i, layer.name, layer.output_shape)
        #
        # image = cv2.imread(os.path.join(trainingpath, 'malignant/ISIC_0027089.jpg'))
        #
        # # image /= 255.0
        # image = np.expand_dims(image, axis=0)
        #
        # feature_maps = model.predict(image)
        #
        # square = 8
        # ix = 1
        # for _ in range(square):
        #     for _ in range(square):
        #     	# specify subplot and turn of axis
        #     	ax = plt.subplot(square, square, ix)
        #     	ax.set_xticks([])
        #     	ax.set_yticks([])
        #     	# plot filter channel in grayscale
        #     	plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
        #     	ix += 1
        # # show the figure
        # plt.show()

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
        preds2 = model.predict_generator(gen_validation, verbose=1)

        # preds is an array like [[x] [x] [x]], make it into array like [x x x]
        preds2 = np.asarray([label for sublist in preds2 for label in sublist])

        # get true validation labels
        true_labels_val2 = gen_validation.classes

        # initialize TPR, FPR, ACC and AUC lists
        TPR_list2, FPR_list2, ACC_list2 = [], [], []
        AUC_score2 = []

        # calculate for different thresholds
        thresholds2 = -np.sort(-(np.unique(preds2)))
        for threshold in thresholds2:
            # apply threshold to predictions
            pred_labels2 = np.where(preds2 > threshold, 1, 0).astype(int)

            # calculate True Positive (TP), True Negative (TN), False Positive (FP) and
            # False Negative (FN)
            TP = np.sum(np.logical_and(pred_labels2 == 1, true_labels_val2 == 1))
            TN = np.sum(np.logical_and(pred_labels2 == 0, true_labels_val2 == 0))
            FP = np.sum(np.logical_and(pred_labels2 == 1, true_labels_val2 == 0))
            FN = np.sum(np.logical_and(pred_labels2 == 0, true_labels_val2 == 1))

            # calculate TPR, FPR, ACC and add to lists
            TPR = TP / (TP + FN)
            FPR = FP / (FP + TN)
            ACC = (TP + TN) / (TP + TN + FP + FN)

            TPR_list2.append(TPR)
            FPR_list2.append(FPR)
            ACC_list2.append(ACC)

            AUC_score2.append((1-FPR+TPR)/2)

            pred_labels2 = []

        AUC_ft = round(sum(AUC_score2)/len(thresholds2),3)
        print("AUC: {}".format(AUC_ft))

        pred_labels2 = np.where(preds2 > 0.5, 1, 0).astype(int)

        # now with sklearn implementation
        fpr, tpr, thresholds2 = roc_curve(true_labels_val2, preds2, pos_label=1)

        AUC2_ft = round(roc_auc_score(true_labels_val2, preds2),3)
        print("sk_AUC: {}".format(AUC2_ft))

        # calculate True Positive (TP), True Negative (TN), False Positive (FP) and
        # False Negative (FN)
        TP = np.sum(np.logical_and(pred_labels2 == 1, true_labels_val2 == 1))
        TN = np.sum(np.logical_and(pred_labels2 == 0, true_labels_val2 == 0))
        FP = np.sum(np.logical_and(pred_labels2 == 1, true_labels_val2 == 0))
        FN = np.sum(np.logical_and(pred_labels2 == 0, true_labels_val2 == 1))

        ACC_ft = round(((TP + TN) / (TP + TN + FP + FN)),3)

        print(ACC_ft)


        # get path to save csv file for results
        FT_csvpath = os.path.join(resultspath, 'FT_results.csv')
        print(FT_csvpath)

        # create csv file if it doesn't exist yet with the correct headers
        if not os.path.exists(FT_csvpath):
            # initialize dataframe to save results for different combinations of datasets and add to csv file
            FT_results = pd.DataFrame(columns=['target_dataset', 'source_dataset', 'AUC', 'skl_AUC', 'ACC', 'AUC_ft', 'skl_AUC_ft', 'ACC_ft'])
            FT_results.to_csv(FT_csvpath, index=False)

        # add new results to dataframe
        row = pd.Series({'target_dataset': args['dataset'],
                        'source_dataset': args['source_dataset'],
                        'AUC': AUC,
                        'skl_AUC': AUC2,
                        'ACC': ACC1,
                        'AUC_ft': AUC_ft,
                        'skl_AUC_ft': AUC2_ft,
                        'ACC_ft': ACC_ft})

        # read existing dataframe, add new row and save again
        FT_results = pd.read_csv(FT_csvpath)
        FT_results = FT_results.append(row, ignore_index=True)
        FT_results.to_csv(FT_csvpath, index=False)






        # # reset the testing generator for network evaluation using the test data
        # print("evaluating after fine-tuning top model...")
        # gen_test.reset()
        #
        # # make predictions and take highest predicted value as class label
        # preds = model.predict_generator(gen_test, steps=(num_test//batchsize), verbose=1)
        # preds = np.argmax(preds, axis=1)
        #
        # # print classification report
        # print(classification_report(gen_test.classes, preds, target_names=gen_test.class_indices.keys()))
        #
        # # create plot directory if it doesn't exist and plot training progress
        # if not os.path.exists(config['plot_path']):
        #     os.makedirs(config['plot_path'])
        # plotpath = os.path.join(config['plot_path'], "warmup_training.png")
        # plot_training(hist, 5, plotpath)
        #
        # # now we can unfreeze base model layers to train more
        # # unfreeze the last convolutional layer in VGG16
        # for layer in base_model.layers[15:]:
        #     layer.trainable = True
        #
        # # print which layers are trainable now
        # for layer in base_model.layers:
        #     print("{}: {}".format(layer, layer.trainable))
        #
        # # reset image generators before training again
        # gen_training.reset()
        # gen_validation.reset()
        #
        # # recompile the model
        # print("recompiling model...")
        # sgd = SGD(lr=1e-4, momentum=0.9)
        # model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
        #
        # # train the model again, with extra trainable layers
        # print("training recompiled model...")
        # hist = model.fit_generator(
        #     gen_training,
        #     steps_per_epoch = num_training // batchsize,
        #     validation_data = gen_validation,
        #     validation_steps = num_validation // batchsize,
        #     epochs=5,
        #     verbose=1)
        #
        # # and evaluate again
        # print("evaluating after fine-tuning network...")
        # gen_test.reset()
        # preds = model.predict_generator(gen_test, steps=(num_test//batchsize), verbose=1)
        # preds = np.argmax(preds, axis=1)
        # print(classification_report(gen_test.classes, preds, target_names=gen_test.class_indices.keys()))
        # plotpath = os.path.join(config['plot_path'], "unfrozen_training.png")
        # plot_training(hist, 5, plotpath)
