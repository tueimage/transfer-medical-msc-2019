import argparse
import os
import json
import glob
import numpy as np
import cv2
import itertools
import random

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
    choices=['balance_down', 'balance_up', 'image_shift_rot'],
    required=True,
    help='modification to apply to dataset')
parser.add_argument('-f',
    '--fraction',
    choices=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
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

# first create necessary directories for modified dataset
for classname in config['classes']:
    # create save directories if they don't exist yet for each class
    train_savedir = os.path.join(trainingpath, classname)
    val_savedir = os.path.join(validationpath, classname)
    test_savedir = os.path.join(testpath, classname)

    train_savedir2 = train_savedir.replace(datasetpath, '{}_{}_f={}'.format(datasetpath, args['modification'], args['fraction']))
    val_savedir2 = val_savedir.replace(datasetpath, '{}_{}_f={}'.format(datasetpath, args['modification'], args['fraction']))
    test_savedir2 = test_savedir.replace(datasetpath, '{}_{}_f={}'.format(datasetpath, args['modification'], args['fraction']))

    if not os.path.exists(train_savedir2):
        os.makedirs(train_savedir2)
    if not os.path.exists(val_savedir2):
        os.makedirs(val_savedir2)
    if not os.path.exists(test_savedir2):
        os.makedirs(test_savedir2)

paths = {}
# separate paths (of training set) for each class in a dictionary
for classname in config['classes']:
    paths[classname] = list(filter(lambda x: classname in x, trainingpaths))

if args['modification'] == 'balance_down':
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
        # for every imagepath in the class
        for imagepath in imagepaths:
            # load image
            image = cv2.imread(imagepath)

            # create a new path to save modified image in
            newpath = imagepath.replace(datasetpath, '{}_{}'.format(datasetpath, args['modification']))

            # save image in the new path
            print("Writing image {} ...".format(newpath))
            cv2.imwrite(newpath, image)

# for the modifications from now on, only fraction of training images is needed
# so we first make a list out of all paths for all keys and shuffle them
allpaths = list(itertools.chain(*paths.values()))
random.shuffle(allpaths)

# and then split the paths into paths to modify and paths not to modify
splitpoint = int(np.ceil(args['fraction']*len(allpaths)))
mod_paths = allpaths[:splitpoint]
rest_paths = allpaths[splitpoint:]

if args['modification'] == 'image_shift_rot':
    # rotates the images with random amounts of rotation
    for imagepath in mod_paths:
        # load image
        image = cv2.imread(imagepath)

        # create a new path to save modified image in
        newpath = imagepath.replace(datasetpath, '{}_{}_f={}'.format(datasetpath, args['modification'], args['fraction']))

        # get the center for rotating the image
        (h, w) = image.shape[:2]
        center = (w/2, h/2)

        # preserve shape of image
        scale = 1.0

        # get a random rotation value from 90, 180 or 270 degrees
        angle = random.sample([90, 180, 270], 1)[0]

        # get rotation matrix and rotate image
        mat = cv2.getRotationMatrix2D(center, angle, scale)
        rot_image = cv2.warpAffine(image, mat, (h, w))

        # save image in the new path
        print("Writing image {} ...".format(newpath))
        cv2.imwrite(newpath, rot_image)

    # save the rest of the images
    for imagepath in rest_paths:
        # do the same, but without modification
        image = cv2.imread(path)
        newpath = imagepath.replace(datasetpath, '{}_{}_f={}'.format(datasetpath, args['modification'], args['fraction']))

        print("Writing image {} ...".format(newpath))
        cv2.imwrite(newpath, image)

















# 1


# always we need to just copy the validation and test set into the new dataset directory
for imagepath in itertools.chain(validationpaths, testpaths):
    # load image
    image = cv2.imread(imagepath)

    # create a new path to save modified image in
    newpath = imagepath.replace(datasetpath, '{}_{}_f={}'.format(datasetpath, args['modification'], args['fraction']))

    # save image in the new path
    print("Writing image {} ...".format(newpath))
    cv2.imwrite(newpath, image)
