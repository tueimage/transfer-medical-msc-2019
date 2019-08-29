from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers import Input
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import TensorBoard, EarlyStopping
from keras import backend as K
import tensorflow as tf
import json
import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import glob
import models
import datetime
import random
import pandas as pd
import cv2
import gc
from time import time
from helper_functions import ROC_AUC, load_data, load_training_data, plot_training, train_model, limit_memory

# choose GPU for training
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# construct argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m',
    '--mode',
    choices=['from_scratch', 'feature_extraction', 'fine_tuning', 'evaluate', 'random_search'],
    required=True,
    help='training mode')
parser.add_argument('-d',
    '--dataset',
    choices=['isic_2017', 'isic_2017_adj'],
    required=True,
    help='dataset to use')
parser.add_argument('-i',
    '--input',
    required='evaluate' in sys.argv,
    help='name of trained model to load when evaluating')
parser.add_argument('-bs', '--batchsize', default=1, help='batch size')
args = vars(parser.parse_args())

# read parameters for wanted dataset from config file
with open('config.json') as json_file:
    config = json.load(json_file)[args['dataset']]

# assign batch size
batchsize = int(args['batchsize'])

# function for randomized search for optimal hyperparameters
if args['mode'] == 'random_search':
    # set a random seed
    sd = 28

    # initialize dataframe to save results for different hyperparameters
    search_records = pd.DataFrame(columns=['epochs', 'train_time', 'AUC', 'skl_AUC', 'train_accuracy', 'val_accuracy', 'learning_rate', 'dropout_rate', 'l2_rate', 'batchsize'])

    trainingpath = config['trainingpath']
    validationpath = config['validationpath']

    # initialize image data generator objects
    gen_obj_training = ImageDataGenerator(rescale=1./255, featurewise_center=True)
    gen_obj_test = ImageDataGenerator(rescale=1./255, featurewise_center=True)

    # we need to fit generators to training data
    # from this mean and std, featurewise_center is calculated in the generator
    x_train = load_training_data(trainingpath)
    gen_obj_training.fit(x_train, seed=sd)
    gen_obj_test.fit(x_train, seed=sd)

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
                learning_rate = 10 ** np.random.uniform(-6,1)
                dropout_rate = np.random.uniform(0,1)
                l2_rate = 10 ** np.random.uniform(-2,-.3)
                batchsize = 2 ** np.random.randint(6)

                # now set a constant random seed, so things that may be variable are the same for every trained model, e.g. weight initialization
                os.environ['PYTHONHASHSEED'] = str(sd)
                np.random.seed(sd)
                random.seed(sd)
                tf.set_random_seed(sd)

                # build the model and train model using the given hyperparameters
                hist, AUC, AUC2, train_time, training_epochs, timestamp = train_model(config, learning_rate, dropout_rate, l2_rate, batchsize, gen_obj_training, gen_obj_test)

                # get train acc and val acc from training history
                validation_acc = hist.history.get('val_acc')[-1]
                train_acc = hist.history.get('acc')[-1]

                # add results to the dataframe
                row = pd.Series({'epochs': training_epochs,
                                'train_time': train_time,
                                'AUC': AUC,
                                'skl_AUC': AUC2,
                                'train_accuracy': train_acc,
                                'val_accuracy': validation_acc,
                                'learning_rate': learning_rate,
                                'dropout_rate': dropout_rate,
                                'l2_rate': l2_rate,
                                'batchsize': batchsize}, name=timestamp)
                search_records = search_records.append(row)
        limit_memory()

    # when searching is done, save dataframe as csv
    csvpath = os.path.join(config['model_savepath'], 'randomsearch.csv')
    search_records.to_csv(csvpath)


# train model from scratch
if args['mode'] == 'from_scratch':
    # get timestamp for saving stuff
    timestamp = datetime.datetime.now().strftime("%y%m%d_%Hh%M")

    # set random seed for result reproducability
    sd=28
    os.environ['PYTHONHASHSEED'] = str(sd)
    np.random.seed(sd)
    random.seed(sd)
    tf.set_random_seed(sd)

    # set input tensor for VGG16 model
    input_tensor = Input(shape=(224,224,3))

    # load VGG16 model architecture
    model_VGG16 = models.model_VGG16(input_tensor)
    print(model_VGG16.summary())

    # set optimizer and compile model
    print("compiling model...")
    sgd = SGD(lr=1e-6, momentum=0.9, nesterov=True)
    RMSprop = RMSprop(lr=1e-6)
    adam = Adam(lr=1e-2, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model_VGG16.compile(loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"])

    tensorboard = TensorBoard(log_dir="logs/{}".format(time()), batch_size=batchsize, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

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
        steps_per_epoch = num_training // batchsize,
        validation_data = gen_validation,
        validation_steps = num_validation // batchsize,
        # class_weight=class_weights,
        epochs=50,
        verbose=1,
        callbacks=[tensorboard])

    # save history
    pd.DataFrame(hist.history).to_csv(os.path.join(config['model_savepath'], '{}_history.csv'.format(timestamp)))

    # create save directory if it doesn't exist and save trained model
    print("saving model...")
    if not os.path.exists(config['model_savepath']):
        os.makedirs(config['model_savepath'])
    savepath = os.path.join(config['model_savepath'], "{}_model_VGG16.h5".format(timestamp))
    model_VGG16.save(savepath)

    # create plot directory if it doesn't exist and plot training progress
    print("saving plots...")
    if not os.path.exists(config['plot_path']):
        os.makedirs(config['plot_path'])
    plotpath = os.path.join(config['plot_path'], "{}_training.png".format(timestamp))
    plot_training(hist, 50, plotpath)

    # check the model on the validation data and use this for tweaking (not on test data)
    # this is for checking the best training settings; afterwards we can test on test set
    print("evaluating model...")

    # make predictions
    preds = model_VGG16.predict_generator(gen_validation, steps=(num_validation//batchsize), verbose=1)

    # get true labels
    true_labels = gen_validation.classes

    # plot ROC and calculate AUC
    ROC_AUC(preds, true_labels, config, timestamp)

if args['mode'] == 'evaluate':
    # load model
    print("loading model...")
    model_VGG16 = load_model(os.path.join(config['model_savepath'], args['input']))

    # if evaluating from saved model, timestamp is retrieved from saved model's
    # name so saved plots will have same timestamp in name as trained model
    timestamp = args['input'][:12]

    # check the model on the validation data and use this for tweaking (not on test data)
    # this is for checking the best training settings; afterwards we can test on test set
    print("evaluating model...")

    # make predictions
    preds = model_VGG16.predict_generator(gen_validation, steps=(num_validation//batchsize), verbose=1)

    # get true labels
    true_labels = gen_validation.classes

    # plot ROC and calculate AUC
    ROC_AUC(preds, true_labels, config, timestamp)

if args['mode'] == 'feature_extraction':
    # get paths to training and test csv files
    trainingpath = os.path.join(config_ISIC.BASE_CSV_PATH, "{}.csv".format(config_ISIC.TRAIN))
    testpath = os.path.join(config_ISIC.BASE_CSV_PATH, "{}.csv".format(config_ISIC.TEST))

    # load data from disk
    print("loading data...")
    (Xtrain, Ytrain) = load_data(trainingpath)
    (Xtest, Ytest) = load_data(testpath)

    # load label encoder
    labelencoder = pickle.loads(open(config_ISIC.LE_PATH, 'rb').read())

    # train the model
    print("training model...")
    model = LogisticRegression(solver="liblinear", multi_class="auto")
    model.fit(Xtrain, Ytrain)

    # evaluate model
    print("evaluating model...")
    preds = model.predict(Xtest)
    print(classification_report(Ytest, preds, target_names=labelencoder.classes_))

    # save model
    print("saving model...")
    with open(config_ISIC.MODEL_PATH, 'wb') as model_file:
        model_file.write(pickle.dumps(model))

if args['mode'] == 'fine_tuning':
    # get paths to training, validation and testing directories
    trainingpath = config['trainingpath']
    validationpath = config['validationpath']
    testpath = config['testpath']

    # get total number of images in each split, needed to train in batches
    num_training = len(glob.glob(os.path.join(trainingpath, '**/*.jpg')))
    num_validation = len(glob.glob(os.path.join(validationpath, '**/*.jpg')))
    num_test = len(glob.glob(os.path.join(testpath, '**/*.jpg')))

    # initialize image data generator objects
    gen_obj_training = ImageDataGenerator()
    gen_obj_test = ImageDataGenerator()

    # add mean subtraction with ImageNet mean to the generator
    imagenet_mean = np.array([123.68, 116.779, 103.939], dtype="float32")
    gen_obj_training.mean = imagenet_mean
    gen_obj_test.mean = imagenet_mean

    # initialize the image generators that load batches of images
    gen_training = gen_obj_training.flow_from_directory(
        trainingpath,
        class_mode="categorical",
        target_size=(224,224),
        color_mode="rgb",
        shuffle=True,
        batch_size=batchsize)

    gen_validation = gen_obj_test.flow_from_directory(
        validationpath,
        class_mode="categorical",
        target_size=(224,224),
        color_mode="rgb",
        shuffle=False,
        batch_size=batchsize)

    gen_test = gen_obj_test.flow_from_directory(
        testpath,
        class_mode="categorical",
        target_size=(224,224),
        color_mode="rgb",
        shuffle=False,
        batch_size=batchsize)

    # now load the VGG16 network with ImageNet weights
    # without last fully connected layer with softmax
    print("loading network...")
    base_model = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224,224,3)))

    # build classifier model to put on top of the base model
    top_model = base_model.output
    top_model = Flatten()(top_model)
    top_model = Dense(32, activation="relu")(top_model)
    top_model = Dropout(0.5)(top_model)
    top_model = Dense(len(config['classes']), activation="softmax")(top_model)

    # add the model on top of the base model
    model = Model(inputs=base_model.input, outputs=top_model)

    # freeze all layers in the base model to exclude them from training
    for layer in base_model.layers:
        layer.trainable = False

    # compile model
    print("compiling model...")
    sgd = SGD(lr=1e-4, momentum=0.9)
    model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])

    # train model (only the top) for a few epochs so the new layers get
    # initialized with learned values instead of randomly
    print("training top model...")
    hist = model.fit_generator(
        gen_training,
        steps_per_epoch = num_training // batchsize,
        validation_data = gen_validation,
        validation_steps = num_validation // batchsize,
        epochs=5,
        verbose=1)

    # reset the testing generator for network evaluation using the test data
    print("evaluating after fine-tuning top model...")
    gen_test.reset()

    # make predictions and take highest predicted value as class label
    preds = model.predict_generator(gen_test, steps=(num_test//batchsize), verbose=1)
    preds = np.argmax(preds, axis=1)

    # print classification report
    print(classification_report(gen_test.classes, preds, target_names=gen_test.class_indices.keys()))

    # create plot directory if it doesn't exist and plot training progress
    if not os.path.exists(config['plot_path']):
        os.makedirs(config['plot_path'])
    plotpath = os.path.join(config['plot_path'], "warmup_training.png")
    plot_training(hist, 5, plotpath)

    # now we can unfreeze base model layers to train more
    # unfreeze the last convolutional layer in VGG16
    for layer in base_model.layers[15:]:
        layer.trainable = True

    # print which layers are trainable now
    for layer in base_model.layers:
        print("{}: {}".format(layer, layer.trainable))

    # reset image generators before training again
    gen_training.reset()
    gen_validation.reset()

    # recompile the model
    print("recompiling model...")
    sgd = SGD(lr=1e-4, momentum=0.9)
    model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])

    # train the model again, with extra trainable layers
    print("training recompiled model...")
    hist = model.fit_generator(
        gen_training,
        steps_per_epoch = num_training // batchsize,
        validation_data = gen_validation,
        validation_steps = num_validation // batchsize,
        epochs=5,
        verbose=1)

    # and evaluate again
    print("evaluating after fine-tuning network...")
    gen_test.reset()
    preds = model.predict_generator(gen_test, steps=(num_test//batchsize), verbose=1)
    preds = np.argmax(preds, axis=1)
    print(classification_report(gen_test.classes, preds, target_names=gen_test.class_indices.keys()))
    plotpath = os.path.join(config['plot_path'], "unfrozen_training.png")
    plot_training(hist, 5, plotpath)
