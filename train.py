from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers import Input
from keras.optimizers import SGD, RMSprop
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
import pandas
import cv2
from evaluate import ROC_AUC

# choose GPU for training
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# set random seed for result reproducability
sd=28
os.environ['PYTHONHASHSEED'] = str(sd)
np.random.seed(sd)
random.seed(sd)
tf.set_random_seed(sd)

# construct argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m',
    '--mode',
    choices=['from_scratch', 'feature_extraction', 'fine_tuning', 'evaluate'],
    required=True,
    help='training mode')
parser.add_argument('-d',
    '--dataset',
    choices=['isic_2017'],
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
batchsize = args['batchsize']

def load_data(splitpath):
    data, labels = [], []

    # loop over the rows in data split file with extracted features
    for row in open(splitpath):
        # extract class label and features and add to lists
        row = row.strip().split(",")
        label = row[0]
        features = np.array(row[1:], dtype="float")

        data.append(features)
        labels.append(label)

    # convert lists to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    return (data, labels)

def plot_training(hist, epochs, plotpath):
    # plot and save training history
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), hist.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), hist.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), hist.history["acc"], label="train_acc")
    plt.plot(np.arange(0, epochs), hist.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper right")
    plt.savefig(plotpath)

def load_training_data(trainingpath):
    # function to load training data
    images = []
    imagepaths = glob.glob(os.path.join(trainingpath, '**/*.jpg'))

    for path in imagepaths:
        images.append(cv2.imread(path))

    images = np.array(images)
    return images


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

# we need to fit generators to training data
# from this mean and std, featurewise_center is calculated in the generator
x_train = load_training_data(trainingpath)
gen_obj_training.fit(x_train)
gen_obj_test.fit(x_train)

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
    batch_size=batchsize)

gen_test = gen_obj_test.flow_from_directory(
    testpath,
    class_mode="binary",
    target_size=(224,224),
    color_mode="rgb",
    shuffle=False,
    batch_size=batchsize)


if args['mode'] == 'from_scratch':
    # get timestamp for saving stuff
    timestamp = datetime.datetime.now().strftime("%y%m%d_%Hh%M")

    # set input tensor for VGG16 model
    input_tensor = Input(shape=(224,224,3))

    # load VGG16 model architecture
    model_VGG16 = models.model_VGG16(input_tensor)
    print(model_VGG16.summary())

    # set optimizer and compile model
    print("compiling model...")
    sgd = SGD(lr=0.01, momentum=0.9)
    RMSprop = RMSprop(lr=1e-6)
    model_VGG16.compile(loss="binary_crossentropy", optimizer=RMSprop, metrics=["accuracy"])

    # # calculate relative class weights for the imbalanced training data
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
        verbose=1)


    # save history
    pandas.DataFrame(hist.history).to_csv(os.path.join(config['model_savepath'], '{}_history.csv'.format(timestamp)))

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
