import argparse
import os
import json
import glob
import numpy as np
import cv2
import itertools

# choose GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# set a seed for reproducability
seed=28

# construct argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d',
    '--dataset',
    choices=['isic_2017'],
    required=True,
    help='dataset to use')
parser.add_argument('-m',
    '--modification',
    choices=['balance_down', 'balance_up'],
    required=True,
    help='modification to apply to dataset')
parser.add_argument('-f',
    '--fraction',
    choices=[0.1, 0.5, 1.0],
    type=float,
    default=1.0,
    help='fraction of total images to apply modification to')
args = vars(parser.parse_args())

# read parameters for wanted dataset from config file
with open('config.json') as json_file:
    config = json.load(json_file)[args['dataset']]

# get dataset path name and paths to image splits
datasetpath = config['dataset_path']
trainingpath = config['trainingpath']
validationpath = config['validationpath']
testpath = config['testpath']

# get imagepaths for each split
trainingpaths = glob.glob(os.path.join(trainingpath, '**/*.jpg'))
validationpaths = glob.glob(os.path.join(validationpath, '**/*.jpg'))
testpaths = glob.glob(os.path.join(testpath, '**/*.jpg'))

# shuffle the training paths and use given fraction of the images, default is all images
np.random.shuffle(trainingpaths)
trainingpaths = trainingpaths[:int(np.ceil(args['fraction']*len(trainingpaths)))]

if args['modification'] == 'balance_down':
    paths = {}
    # separate paths for each class in a dictionary
    for classname in config['classes']:
        paths[classname] = list(filter(lambda x: classname in x, trainingpaths))

    # find class with least images (paths)
    min_paths = min([len(paths[val]) for val in paths.keys()])

    # now give every other class to same amount of paths (images)
    for classname in paths.keys():
        # skip the downsampling for class with least samples
        if len(paths[classname]) == min_paths:
            continue

        # take the first number of paths equal to number of smallest class
        paths[classname] = paths[classname][:min_paths]

    # now do for every class in the dictionary
    for classname, imagepaths in paths.items():
        for every imagepath in the class
        for imagepath in imagepaths:
            # load image
            image = cv2.imread(imagepath)

            # create a new path to save modified image in
            newpath = imagepath.replace(datasetpath, '{}_{}'.format(datasetpath, args['modification']))

            # save image in the new path
            print("Writing image {} ...".format(newpath))
            cv2.imwrite(newpath, image)

    for classname in config['classes']:
        # create save directories if they don't exist yet for each class
        train_savedir = os.path.join(trainingpath, classname)
        val_savedir = os.path.join(validationpath, classname)
        test_savedir = os.path.join(testpath, classname)

        train_savedir2 = train_savedir.replace(datasetpath, '{}_{}'.format(datasetpath, args['modification']))
        val_savedir2 = val_savedir.replace(datasetpath, '{}_{}'.format(datasetpath, args['modification']))
        test_savedir2 = test_savedir.replace(datasetpath, '{}_{}'.format(datasetpath, args['modification']))

        if not os.path.exists(train_savedir2):
            os.makedirs(train_savedir2)
        if not os.path.exists(val_savedir2):
            os.makedirs(val_savedir2)
        if not os.path.exists(test_savedir2):
            os.makedirs(test_savedir2)

    # lastly, also copy validation and test set images, these do no need to be rebalanced
    for imagepath in itertools.chain(validationpaths, testpaths):
        # load image
        image = cv2.imread(imagepath)

        # create a new path to save modified image in
        newpath = imagepath.replace(datasetpath, '{}_{}'.format(datasetpath, args['modification']))

        # save image in the new path
        print("Writing image {} ...".format(newpath))
        cv2.imwrite(newpath, image)








# # find class with most images (paths)
# max_paths = max([len(paths[val]) for val in paths.keys()])
# max_keys = [key for key, val in paths.items() if len(val) == max_paths]

# for max_key in max_keys:
#     print(len(paths[max_key]))
