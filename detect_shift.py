# Code for shift detection between two datasets
import os
import argparse
import json
import sys
import glob
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from models import model_VGG16_light
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from helper_functions import ROC_AUC, plot_training, load_training_data
from sklearn.metrics import accuracy_score

# choose GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# construct argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m',
    '--mode',
    choices=['train_val', 'datasets'],
    required=True,
    help='dataset shift to detect, train_val for shift between train and validation set')
parser.add_argument('-d',
    '--dataset',
    choices=['isic', 'isic_2017', 'cats_and_dogs'],
    required=True,
    help='dataset to use')
parser.add_argument('--d2',
    '--dataset_2',
    required='datasets' in sys.argv,
    help='second dataset to use')

args = vars(parser.parse_args())

# read parameters for wanted dataset from config file
with open('config.json') as json_file:
    config = json.load(json_file)[args['dataset']]

# get training and validation path
trainingpath = config['trainingpath']
validationpath = config['validationpath']

# split both datasets in two
impaths_train = glob.glob(os.path.join(trainingpath, '**/*.jpg'))
impaths_val = glob.glob(os.path.join(validationpath, '**/*.jpg'))

# now split them both in half to use for training and testing
impaths_train_t = impaths_train[:int(np.ceil(0.5*len(impaths_train)))]
impaths_val_t = impaths_val[:int(np.ceil(0.5*len(impaths_val)))]

impaths_train_s = impaths_train[int(np.ceil(0.5*len(impaths_train))):]
impaths_val_s = impaths_val[int(np.ceil(0.5*len(impaths_val))):]

# create arrays with training and validation data
x_train, y_train = [], []
for path in impaths_train_t:
    x_train.append(cv2.imread(path))
    y_train.append(0)
for path in impaths_val_t:
    x_train.append(cv2.imread(path))
    y_train.append(1)

x_train = np.array(x_train)
y_train = np.array(y_train)

x_val, y_val = [], []
for path in impaths_train_s:
    x_val.append(cv2.imread(path))
    y_val.append(0)
for path in impaths_val_s:
    x_val.append(cv2.imread(path))
    y_val.append(1)


x_train = np.array(x_train)
y_train = np.array(y_train)
x_val = np.array(x_val)
y_val = np.array(y_val)

# shuffle the arrays with same seed
sd=28
np.random.seed(sd)
np.random.shuffle(x_train)
np.random.shuffle(y_train)

# get model to train classifier
input_tensor=Input(shape=(224,224,3))
model_VGG16 = model_VGG16_light(input_tensor)

# initialize image data generator objects
gen_obj_training = ImageDataGenerator(rescale=1./255, featurewise_center=True)
gen_obj_test = ImageDataGenerator(rescale=1./255, featurewise_center=True)

# we need to fit generators to training data
# from this mean and std, featurewise_center is calculated in the generator
gen_obj_training.fit(x_train, seed=sd)
gen_obj_test.fit(x_train, seed=sd)


# set optimizer and compile model
learning_rate = 1e-6
print("compiling model...")
sgd = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
model_VGG16.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])

# train the model
print("training model...")
epochs=50
early_stopping = EarlyStopping(monitor='acc', mode='max', patience=10)

# fits the model on batches with real-time data augmentation:
hist = model_VGG16.fit_generator(gen_obj_training.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs, verbose=1, callbacks=[early_stopping])

# create save directories if it doesn't exist
if not os.path.exists(config['model_savepath']):
    os.makedirs(config['model_savepath'])
if not os.path.exists(config['plot_path']):
    os.makedirs(config['plot_path'])

# create a name for the model
if args['mode'] == 'train_val':
    modelname = 'shift_{}_{}'.format(args['mode'], args['dataset'])
if args['mode'] == 'datasets':
    modelname = 'shift_{}_{}'.format(args['dataset'], args['dataset2'])

# save history
pd.DataFrame(hist.history).to_csv(os.path.join(config['model_savepath'], '{}_history.csv'.format(modelname)))

# save trained model
print("saving model...")
savepath = os.path.join(config['model_savepath'], "{}_VGG16.h5".format(modelname))
model_VGG16.save(savepath)

# create plot directory if it doesn't exist and plot training progress
print("saving plots...")
plotpath = os.path.join(config['plot_path'], "{}_training.png".format(modelname))
# plot_training(hist, epochs, plotpath)

# now make predictions
print("evaluating model...")
preds = model_VGG16.predict_generator(gen_obj_test.flow(x_val, batch_size=1), verbose=1)

# get true labels
true_labels = y_val

# plot ROC and calculate AUC
ROC_AUC(preds, true_labels, config, modelname)

predictions = np.where(preds >= 0.5, 1, 0).astype(int)
accuracy = accuracy_score(true_labels, predictions)

print(accuracy)
