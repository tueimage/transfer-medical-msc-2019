# script to build dataset in a generic structure
# structure should be dataset_name/split/class/images.ext

import os
import glob
import cv2
import numpy as np
import argparse
import json

# construct argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d',
    '--dataset',
    choices=['isic', 'isic_2017', 'cats_and_dogs', 'ddsm', 'CNMC'],
    required=True,
    help='dataset to use')
args = vars(parser.parse_args())

# read parameters for wanted dataset from config file
with open('config.json') as json_file:
    config = json.load(json_file)[args['dataset']]

def resize_image(image):
    # find smallest and biggest dimension of the x- and y- dims
    s_dim = np.min((image.shape[0], image.shape[1]))
    m_dim = np.max((image.shape[0], image.shape[1]))

    # randomly crop image to [smallest dimension x smallest dimension]
    offset = np.random.random_integers(0, high=m_dim-s_dim)
    if s_dim == image.shape[0]:
        cropped_image = image[:,offset:offset+s_dim,:]
    if s_dim == image.shape[1]:
        cropped_image = image[offset:offset+s_dim,:,:]

    # now resize the cropped images to 224x224
    resized_image = cv2.resize(cropped_image, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)

    return resized_image

if args['dataset'] == 'CNMC':
    # percentages to split data in training, val and test set
    split_pct = [.8, .1, .1]

    trainingpath = os.path.join(config['orig_data_path'], 'C-NMC_training_data')

    paths_all = glob.glob(os.path.join(trainingpath, '*/all/*.bmp'))
    paths_hem = glob.glob(os.path.join(trainingpath, '*/hem/*.bmp'))

    # get class with least amount of images
    min = min(len(paths_all), len(paths_hem))

    # shuffle all paths
    np.random.shuffle(paths_all)
    np.random.shuffle(paths_hem)

    # get same amount of images for each class
    paths_all = paths_all[:min]
    paths_hem = paths_hem[:min]

    splitnr_train = int(np.ceil(len(paths_all)*split_pct[0]))
    splitnr_validation = int(np.ceil(len(paths_all)*split_pct[1]))
    splitnr_test = int(np.ceil(len(paths_all)*split_pct[2]))

    for imclass in config['classes']:
        if imclass == 'normal':
            paths = paths_hem
        elif imclass == 'leukemic':
            paths = paths_all

        for split in ['training', 'validation', 'test']:
            # create path where images in corresponding split should be saved
            savedir = os.path.join(config['dataset_path'], split)

            if split == 'training':
                split_paths = paths[:splitnr_train]
            if split == 'validation':
                split_paths = paths[splitnr_train:splitnr_train+splitnr_validation]
            if split == 'test':
                split_paths = paths[splitnr_train+splitnr_validation:]

            # create folders to save images in if it doesn't exist yet
            target_path = os.path.join(savedir, imclass)
            if not os.path.exists(target_path):
                os.makedirs(target_path)

            for impath in split_paths:
                # get filename
                filename = os.path.basename(impath)

                # read image
                image = cv2.imread(impath)

                # resize image to 224x224
                resized_image = resize_image(image)

                # save image to right path
                savepath = os.path.join(config['dataset_path'], split, imclass, filename.replace('.bmp', '.jpg'))
                print(savepath)
                cv2.imwrite(savepath, resized_image)

# for isic dataset
if args['dataset'] == 'isic':
    # percentages to split data in training, val and test set
    split_pct = [.8, .1, .1]

    # get all paths for both classes
    allpaths_benign = glob.glob(os.path.join(config['orig_data_path'], config['classes'][0], '*.jpg'))
    allpaths_malignant = glob.glob(os.path.join(config['orig_data_path'], config['classes'][1], '*.jpg'))

    splitnr_train = int(np.ceil(len(allpaths_benign)*split_pct[0]))
    splitnr_validation = int(np.ceil(len(allpaths_benign)*split_pct[1]))
    splitnr_test = int(np.ceil(len(allpaths_benign)*split_pct[2]))

    # split data paths according to splits
    allpaths_benign_train = allpaths_benign[:splitnr_train]
    allpaths_benign_validation = allpaths_benign[splitnr_train:splitnr_train+splitnr_validation]
    allpaths_benign_test = allpaths_benign[splitnr_train+splitnr_validation:]

    allpaths_malignant_train = allpaths_malignant[:splitnr_train]
    allpaths_malignant_validation = allpaths_malignant[splitnr_train:splitnr_train+splitnr_validation]
    allpaths_malignant_test = allpaths_malignant[splitnr_train+splitnr_validation:]

    # print(len(allpaths_benign_train))
    # print(len(allpaths_malignant_train))
    # print(len(allpaths_benign_validation))
    # print(len(allpaths_malignant_validation))
    # print(len(allpaths_benign_test))
    # print(len(allpaths_malignant_test))

    # do for every split
    for split in ('training', 'validation', 'test'):
        # create path where images in corresponding split should be saved
        savedir = os.path.join(config['dataset_path'], split)

        # find path to images to corresponding class
        for imclass in config['classes']:
            # create folders to save images in if it doesn't exist yet
            target_path = os.path.join(savedir, imclass)
            if not os.path.exists(target_path):
                os.makedirs(target_path)

            if split == 'training':
                if imclass == 'benign':
                    for impath in allpaths_benign_train:
                        # get filename
                        filename = os.path.basename(impath)

                        # read image
                        image = cv2.imread(impath)

                        # resize image to 224x224
                        resized_image = resize_image(image)

                        # save image to right path
                        savepath = os.path.join(config['dataset_path'], split, imclass, filename)
                        print(savepath)
                        cv2.imwrite(savepath, resized_image)
                if imclass == 'malignant':
                    for impath in allpaths_malignant_train:
                        # get filename
                        filename = os.path.basename(impath)

                        # read image
                        image = cv2.imread(impath)

                        # resize image to 224x224
                        resized_image = resize_image(image)

                        # save image to right path
                        savepath = os.path.join(config['dataset_path'], split, imclass, filename)
                        cv2.imwrite(savepath, resized_image)
            if split == 'validation':
                if imclass == 'benign':
                    for impath in allpaths_benign_validation:
                        # get filename
                        filename = os.path.basename(impath)

                        # read image
                        image = cv2.imread(impath)

                        # resize image to 224x224
                        resized_image = resize_image(image)

                        # save image to right path
                        savepath = os.path.join(config['dataset_path'], split, imclass, filename)
                        cv2.imwrite(savepath, resized_image)

                if imclass == 'malignant':
                    for impath in allpaths_malignant_validation:
                        # get filename
                        filename = os.path.basename(impath)

                        # read image
                        image = cv2.imread(impath)

                        # resize image to 224x224
                        resized_image = resize_image(image)

                        # save image to right path
                        savepath = os.path.join(config['dataset_path'], split, imclass, filename)
                        cv2.imwrite(savepath, resized_image)
            if split == 'test':
                if imclass == 'benign':
                    for impath in allpaths_benign_test:
                        # get filename
                        filename = os.path.basename(impath)

                        # read image
                        image = cv2.imread(impath)

                        # resize image to 224x224
                        resized_image = resize_image(image)

                        # save image to right path
                        savepath = os.path.join(config['dataset_path'], split, imclass, filename)
                        cv2.imwrite(savepath, resized_image)

                if imclass == 'malignant':
                    for impath in allpaths_malignant_test:
                        # get filename
                        filename = os.path.basename(impath)

                        # read image
                        image = cv2.imread(impath)

                        # resize image to 224x224
                        resized_image = resize_image(image)

                        # save image to right path
                        savepath = os.path.join(config['dataset_path'], split, imclass, filename)
                        cv2.imwrite(savepath, resized_image)


# for isic_2017 dataset
if args['dataset'] == 'isic_2017':
    # do for every split
    for split in ('training', 'validation', 'test'):
        print("processing '{} split'...".format(split))

        # create path where images in corresponding split should be saved
        savedir = os.path.join(config['dataset_path'], split)

        # find path to images corresponding to split
        imdir = glob.glob(os.path.join(config['orig_data_path'], '*{}*Data'.format(split)))[0]

        # find path to ground truth csv file
        gt_path = glob.glob(os.path.join(config['orig_data_path'], '*{}*.csv'.format(split)))[0]

        # create paths for output directories if they don't exist yet
        target_impath_0 = os.path.join(savedir, 'nevus_sk')
        if not os.path.exists(target_impath_0):
            os.makedirs(target_impath_0)

        target_impath_1 = os.path.join(savedir, 'melanoma')
        if not os.path.exists(target_impath_1):
            os.makedirs(target_impath_1)

        # now use csv file to find filenames for images in corresponding split
        with open(gt_path) as gt_csv:
            next(gt_csv)
            for row in gt_csv:
                row = row.strip().split(",")
                im = row[0]
                label1 = row[1]
                label2 = row[2]

                # get image file path and read image
                impath = os.path.join(imdir, '{}.jpg'.format(im))
                image = cv2.imread(impath)

                # resize image to 224x224
                resized_image = resize_image(image)

                if int(float(label1)) == 0:
                    # save images which belong to nevus or seborrheic keratosis class (label 0)
                    cv2.imwrite(os.path.join(target_impath_0, '{}.jpg'.format(im)), resized_image)

                elif int(float(label1)) == 1:
                    # save images which belong to nevus or seborrheic keratosis class (label 0)
                    cv2.imwrite(os.path.join(target_impath_1, '{}.jpg'.format(im)), resized_image)

# for dataset cats_and_dogs
if args['dataset'] == 'cats_and_dogs':
    # percentages to split data in training, val and test set
    split_pct = [.8, .1, .1]

    # get all paths for dog images
    allpaths_cat = glob.glob(os.path.join(config['orig_data_path'], 'PetImages/Cat/*.jpg'))
    allpaths_dog = glob.glob(os.path.join(config['orig_data_path'], 'PetImages/Dog/*.jpg'))

    splitnr_train = int(np.ceil(len(allpaths_cat)*split_pct[0]))
    splitnr_validation = int(np.ceil(len(allpaths_cat)*split_pct[1]))
    splitnr_test = int(np.ceil(len(allpaths_cat)*split_pct[2]))

    # split data paths according to splits
    allpaths_cat_train = allpaths_cat[:splitnr_train]
    allpaths_cat_validation = allpaths_cat[splitnr_train:splitnr_train+splitnr_validation]
    allpaths_cat_test = allpaths_cat[splitnr_train+splitnr_validation:]

    allpaths_dog_train = allpaths_dog[:splitnr_train]
    allpaths_dog_validation = allpaths_dog[splitnr_train:splitnr_train+splitnr_validation]
    allpaths_dog_test = allpaths_dog[splitnr_train+splitnr_validation:]

    # do for every split
    for split in ('training', 'validation', 'test'):
        # create path where images in corresponding split should be saved
        savedir = os.path.join(config['dataset_path'], split)

        # find path to images to corresponding class
        for imclass in config['classes']:
            # create folders to save images in if it doesn't exist yet
            target_path = os.path.join(savedir, imclass)
            if not os.path.exists(target_path):
                os.makedirs(target_path)

            if split == 'training':
                if imclass == 'cat':
                    for impath in allpaths_cat_train:
                        # get filename
                        filename = os.path.basename(impath)

                        # read image
                        image = cv2.imread(impath)

                        # resize image to 224x224
                        resized_image = resize_image(image)

                        # save image to right path
                        savepath = os.path.join(config['dataset_path'], split, imclass, filename)
                        print(savepath)
                        cv2.imwrite(savepath, resized_image)
                if imclass == 'dog':
                    for impath in allpaths_dog_train:
                        # get filename
                        filename = os.path.basename(impath)

                        # read image
                        image = cv2.imread(impath)

                        # resize image to 224x224
                        resized_image = resize_image(image)

                        # save image to right path
                        savepath = os.path.join(config['dataset_path'], split, imclass, filename)
                        cv2.imwrite(savepath, resized_image)
            if split == 'validation':
                if imclass == 'cat':
                    for impath in allpaths_cat_validation:
                        # get filename
                        filename = os.path.basename(impath)

                        # read image
                        image = cv2.imread(impath)

                        # resize image to 224x224
                        resized_image = resize_image(image)

                        # save image to right path
                        savepath = os.path.join(config['dataset_path'], split, imclass, filename)
                        cv2.imwrite(savepath, resized_image)

                if imclass == 'dog':
                    for impath in allpaths_dog_validation:
                        # get filename
                        filename = os.path.basename(impath)

                        # read image
                        image = cv2.imread(impath)

                        # resize image to 224x224
                        resized_image = resize_image(image)

                        # save image to right path
                        savepath = os.path.join(config['dataset_path'], split, imclass, filename)
                        cv2.imwrite(savepath, resized_image)
            if split == 'test':
                if imclass == 'cat':
                    for impath in allpaths_cat_test:
                        # get filename
                        filename = os.path.basename(impath)

                        # read image
                        image = cv2.imread(impath)

                        # resize image to 224x224
                        resized_image = resize_image(image)

                        # save image to right path
                        savepath = os.path.join(config['dataset_path'], split, imclass, filename)
                        cv2.imwrite(savepath, resized_image)

                if imclass == 'dog':
                    for impath in allpaths_dog_test:
                        # get filename
                        filename = os.path.basename(impath)

                        # read image
                        image = cv2.imread(impath)

                        # resize image to 224x224
                        resized_image = resize_image(image)

                        # save image to right path
                        savepath = os.path.join(config['dataset_path'], split, imclass, filename)
                        cv2.imwrite(savepath, resized_image)
