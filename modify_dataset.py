import argparse
import os
import json
import glob
import numpy as np
import cv2
import itertools
import random
import sys
import keras
import pickle
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from utils import *
from main import NeuralNetwork
from scipy.stats import entropy
import time
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# choose GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# set a seed for reproducability
seed=28

# construct argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d',
    '--dataset',
    choices=['ISIC_2', 'ISIC_3', 'ISIC_4', 'ISIC_5', 'ISIC_6', 'CNMC_2', 'CNMC_3', 'CNMC_4', 'CNMC_5', 'CNMC_6'],
    required=True,
    help='dataset to use')
parser.add_argument('-m',
    '--modification',
    choices=['change_split', 'balance_down', 'balance_up', 'image_rot', 'image_translation', 'image_zoom', 'add_noise', 'imbalance_classes', 'grayscale', 'hsv', 'blur', 'small_random', 'small_easy', 'small_hard', 'small_clusters'],
    required=True,
    help='modification to apply to dataset')
parser.add_argument('-n',
    '--noise',
    choices=['gaussian', 'poisson', 'salt_and_pepper', 'speckle'],
    required='add_noise' in sys.argv,
    help='noise type to include to dataset')
parser.add_argument('-f',
    '--fraction',
    choices=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    type=float,
    default=1.0,
    help='fraction of total images to apply modification to')
parser.add_argument('-s',
    '--seed',
    required='change_split' in sys.argv,
    help='seed to re-split the dataset with')
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


    if args['modification'] == 'add_noise':
        train_savedir2 = train_savedir.replace(datasetpath, '{}_{}_{}_f={}'.format(datasetpath, args['modification'], args['noise'], args['fraction']))
        val_savedir2 = val_savedir.replace(datasetpath, '{}_{}_{}_f={}'.format(datasetpath, args['modification'], args['noise'], args['fraction']))
        test_savedir2 = test_savedir.replace(datasetpath, '{}_{}_{}_f={}'.format(datasetpath, args['modification'], args['noise'], args['fraction']))

    elif args['modification'] == 'change_split':
        seed = int(args['seed'])
        train_savedir2 = train_savedir.replace(datasetpath, '{}_{}'.format(datasetpath, seed))
        val_savedir2 = val_savedir.replace(datasetpath, '{}_{}'.format(datasetpath, seed))
        test_savedir2 = test_savedir.replace(datasetpath, '{}_{}'.format(datasetpath, seed))

    else:
        train_savedir2 = train_savedir.replace(datasetpath, '{}_{}_f={}'.format(datasetpath, args['modification'], args['fraction']))
        val_savedir2 = val_savedir.replace(datasetpath, '{}_{}_f={}'.format(datasetpath, args['modification'], args['fraction']))
        test_savedir2 = test_savedir.replace(datasetpath, '{}_{}_f={}'.format(datasetpath, args['modification'], args['fraction']))

    if not os.path.exists(train_savedir2):
        os.makedirs(train_savedir2)
    if not os.path.exists(val_savedir2):
        os.makedirs(val_savedir2)
    if not os.path.exists(test_savedir2):
        os.makedirs(test_savedir2)

paths, paths_val, paths_test = {}, {}, {}
# separate paths (of training set) for each class in a dictionary
for classname in config['classes']:
    paths[classname] = list(filter(lambda x: classname in x, trainingpaths))

if args['modification'] == 'change_split':
    # also read all validation images and test images per class
    for classname in config['classes']:
        paths_val[classname] = list(filter(lambda x: classname in x, validationpaths))
        paths_test[classname] = list(filter(lambda x: classname in x, testpaths))

    # concatenate all paths from each split
    path_dicts = [paths, paths_val, paths_test]
    allpaths = {}
    for key in paths.keys():
        allpaths[key] = np.concatenate(list(dict[key] for dict in [paths, paths_val, paths_test]))

    # now we need to shuffle the values for all classes
    np.random.seed(seed)
    for key,  val in allpaths.items():
        np.random.shuffle(val)
        allpaths[key] = val

    # percentages to split data in training, val and test set
    split_pct = {'training': .8, 'validation': .1, 'test':.1}

    # split the shuffled images and save in right directories
    for key, val in allpaths.items():
        k = 0
        for split in ['training', 'validation', 'test']:
            # split the paths
            splitnr = int(np.ceil(len(val)*split_pct[split]))
            splitpaths = val[k:k+splitnr]

            # use a counter so same images are not re-used in next split
            k += splitnr
            for imagepath in splitpaths:
                # load image
                image = cv2.imread(imagepath)

                # first replace the split with the correct split, try for all splits
                for pos_split in ['training', 'validation', 'test']:
                    imagepath = imagepath.replace(pos_split, split)

                # also replace dataset name
                newpath = imagepath.replace(datasetpath, '{}_{}'.format(datasetpath, seed))

                # save image in the new path
                print("Writing image {} ...".format(newpath))
                cv2.imwrite(newpath, image)

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
            newpath = imagepath.replace(datasetpath, '{}_{}_f={}'.format(datasetpath, args['modification'], args['fraction']))

            # save image in the new path
            print("Writing image {} ...".format(newpath))
            cv2.imwrite(newpath, image)

if args['modification'] == 'balance_up':

    # find class with least images (paths)
    max_paths = max([len(paths[val]) for val in paths.keys()])

    # now give every other class the same amount of paths (images)
    for classname in paths.keys():
        # skip the downsampling for class with least samples
        if len(paths[classname]) == max_paths:
            continue

        # duplicate random images to get the same amount as the biggest class
        extra_paths = [random.choice(paths[classname]) for _ in range(max_paths-len(paths[classname]))]
        extra_paths_orig = np.copy(extra_paths)

        # change the name of the duplicates so the image are not just replaced
        for i in range(len(extra_paths)):
            path = extra_paths[i].replace('.jpg', '_{}.jpg'.format(i))
            extra_paths[i] = path

        # add the original images to new dataset
        for imagepath in paths[classname]:
            # load image
            image = cv2.imread(imagepath)

            # create a new path to save modified image in
            newpath = imagepath.replace(datasetpath, '{}_{}_f={}'.format(datasetpath, args['modification'], args['fraction']))

            # save image in the new path
            print("Writing image {} ...".format(newpath))
            cv2.imwrite(newpath, image)

        # add the extra images to new dataset
        for i in range(len(extra_paths)):
            # load original image
            image = cv2.imread(extra_paths_orig[i])

            # create new path
            newpath = extra_paths[i].replace(datasetpath, '{}_{}_f={}'.format(datasetpath, args['modification'], args['fraction']))

            # save the duplicate image with the name new in new path
            print("Writing image {} ...".format(newpath))
            cv2.imwrite(newpath, image)

if args['modification'] == 'imbalance_classes':
    if args['dataset'] in ['ISIC_2', 'ISIC_3', 'ISIC_4', 'ISIC_5', 'ISIC_6']:
        benign_paths = paths['benign']
        malignant_paths = paths['malignant']

    if args['dataset'] in ['CNMC_2', 'CNMC_3', 'CNMC_4', 'CNMC_5', 'CNMC_6']:
        benign_paths = paths['normal']
        malignant_paths = paths['leukemic']

    malignant_paths = malignant_paths[:int(np.ceil(args['fraction']*len(malignant_paths)))]

    for imagepath in itertools.chain(benign_paths, malignant_paths):
        # load image
        image = cv2.imread(imagepath)

        # create a new path to save modified image in
        newpath = imagepath.replace(datasetpath, '{}_{}_f={}'.format(datasetpath, args['modification'], args['fraction']))

        # save image in the new path
        print("Writing image {} ...".format(newpath))
        cv2.imwrite(newpath, image)

if args['modification'] == 'small_random':
    if args['dataset'] in ['ISIC_2', 'ISIC_3', 'ISIC_4', 'ISIC_5', 'ISIC_6']:
        benign_paths = paths['benign']
        malignant_paths = paths['malignant']

    if args['dataset'] in ['CNMC_2', 'CNMC_3', 'CNMC_4', 'CNMC_5', 'CNMC_6']:
        benign_paths = paths['normal']
        malignant_paths = paths['leukemic']

    malignant_paths = malignant_paths[:int(np.ceil(args['fraction']*len(malignant_paths)))]
    benign_paths = benign_paths[:int(np.ceil(args['fraction']*len(benign_paths)))]

    for imagepath in itertools.chain(benign_paths, malignant_paths):
        # load image
        image = cv2.imread(imagepath)

        # create a new path to save modified image in
        newpath = imagepath.replace(datasetpath, '{}_{}_f={}'.format(datasetpath, args['modification'], args['fraction']))

        # save image in the new path
        print("Writing image {} ...".format(newpath))
        cv2.imwrite(newpath, image)

if args['modification'] == 'small_hard':
    # first load the trained network
    print("loading source network...")
    modelpath = os.path.join(config['model_savepath'], '{}_model.h5'.format(args['dataset']))
    model = load_model(modelpath)
    model.summary()

    # initialize image data generator objects
    gen_obj_training = ImageDataGenerator(rescale=1./255, featurewise_center=True)

    # fit generator to training data
    x_train = load_training_data(trainingpath)
    gen_obj_training.fit(x_train, seed=seed)

    # initialize image generators that load batches of images
    # no shuffle to get pathnames in the same order as the predictions
    gen_training = gen_obj_training.flow_from_directory(
        trainingpath,
        class_mode="binary",
        target_size=(224,224),
        color_mode="rgb",
        shuffle=False,
        batch_size=1)

    # get predictions
    predictions = model.predict_generator(gen_training, verbose=1)

    # preds is an array like [[x] [x] [x]], make it into array like [x x x]
    predictions = np.asarray([label for sublist in predictions for label in sublist])

    # the filenames from the generator are for both classes
    # we want to split them up, along with the prediction values to keep the class balance
    benign_gen_paths, malignant_gen_paths, benign_preds, malignant_preds = [], [], [], []
    for i, filename in enumerate(gen_training.filenames):
        if args['dataset'] in ['ISIC_2', 'ISIC_3', 'ISIC_4', 'ISIC_5', 'ISIC_6']:
            if "malignant" in filename:
                malignant_gen_paths.append(filename)
                malignant_preds.append(predictions[i])
            if "benign" in filename:
                benign_gen_paths.append(filename)
                benign_preds.append(predictions[i])
        if args['dataset'] in ['CNMC_2', 'CNMC_3', 'CNMC_4', 'CNMC_5', 'CNMC_6']:
            if "leukemic" in filename:
                malignant_gen_paths.append(filename)
                malignant_preds.append(predictions[i])
            if "normal" in filename:
                benign_gen_paths.append(filename)
                benign_preds.append(predictions[i])

    # sort both the paths and the filenames in the same way using sortedzip
    szip_benign = sorted(zip(benign_preds, benign_gen_paths))
    szip_malignant = sorted(zip(malignant_preds, malignant_gen_paths))

    # create a list of only the image names, these will now be ordered in the order of the corresponding prediction values
    benign_paths_sorted = [label for pred,label in szip_benign]
    malignant_paths_sorted = [label for pred,label in szip_malignant]

    # now find out how many image names should be removed from both sides of the list
    # both sides because lost and highest prediction values correspond to most confident predictions
    toremove = int(np.ceil((len(benign_paths_sorted) - int(np.ceil(args['fraction']*len(benign_paths_sorted))))/2))

    # if f=1.0 we dont need to remove anything
    if toremove == 0:
        benign_paths_small = benign_paths_sorted
        malignant_paths_small = malignant_paths_sorted
    else:
        # remove the filenames
        benign_paths_small = benign_paths_sorted[toremove:-toremove]
        malignant_paths_small = malignant_paths_sorted[toremove:-toremove]

    # get the full pathnames
    benign_paths_full = [os.path.join(config['trainingpath'], path) for path in benign_paths_small]
    malignant_paths_full = [os.path.join(config['trainingpath'], path) for path in malignant_paths_small]

    # now save all these images in a new directory
    for imagepath in itertools.chain(benign_paths_full, malignant_paths_full):
        # load image
        image = cv2.imread(imagepath)

        # create a new path to save modified image in
        newpath = imagepath.replace(datasetpath, '{}_{}_f={}'.format(datasetpath, args['modification'], args['fraction']))

        # save image in the new path
        print("Writing image {} ...".format(newpath))
        cv2.imwrite(newpath, image)

if args['modification'] == 'small_easy':
    # first load the trained network
    print("loading source network...")
    modelpath = os.path.join(config['model_savepath'], '{}_model.h5'.format(args['dataset']))
    model = load_model(modelpath)
    model.summary()

    # initialize image data generator objects
    gen_obj_training = ImageDataGenerator(rescale=1./255, featurewise_center=True)

    # fit generator to training data
    x_train = load_training_data(trainingpath)
    gen_obj_training.fit(x_train, seed=seed)

    # initialize image generators that load batches of images
    # no shuffle to get pathnames in the same order as the predictions
    gen_training = gen_obj_training.flow_from_directory(
        trainingpath,
        class_mode="binary",
        target_size=(224,224),
        color_mode="rgb",
        shuffle=False,
        batch_size=1)

    # get predictions
    predictions = model.predict_generator(gen_training, verbose=1)

    # preds is an array like [[x] [x] [x]], make it into array like [x x x]
    predictions = np.asarray([label for sublist in predictions for label in sublist])

    # the filenames from the generator are for both classes
    # we want to split them up, along with the prediction values to keep the class balance
    benign_gen_paths, malignant_gen_paths, benign_preds, malignant_preds = [], [], [], []
    for i, filename in enumerate(gen_training.filenames):
        if args['dataset'] in ['ISIC_2', 'ISIC_3', 'ISIC_4', 'ISIC_5', 'ISIC_6']:
            if "malignant" in filename:
                malignant_gen_paths.append(filename)
                malignant_preds.append(predictions[i])
            if "benign" in filename:
                benign_gen_paths.append(filename)
                benign_preds.append(predictions[i])
        if args['dataset'] in ['CNMC_2', 'CNMC_3', 'CNMC_4', 'CNMC_5', 'CNMC_6']:
            if "leukemic" in filename:
                malignant_gen_paths.append(filename)
                malignant_preds.append(predictions[i])
            if "normal" in filename:
                benign_gen_paths.append(filename)
                benign_preds.append(predictions[i])

    # sort both the paths and the filenames in the same way using sortedzip
    szip_benign = sorted(zip(benign_preds, benign_gen_paths))
    szip_malignant = sorted(zip(malignant_preds, malignant_gen_paths))

    # create a list of only the image names, these will now be ordered in the order of the corresponding prediction values
    benign_paths_sorted = [label for pred,label in szip_benign]
    malignant_paths_sorted = [label for pred,label in szip_malignant]

    # now find out how many image names should be removed from both sides of the list
    # both sides because lowest and highest prediction values correspond to most confident predictions
    toremove = int(np.ceil((len(benign_paths_sorted) - int(np.ceil(args['fraction']*len(benign_paths_sorted))))/2))
    # not completely right, 0.1 will be 0.9, 0.2 will be 0.8 etc..

    benign_paths_small = benign_paths_sorted
    malignant_paths_small = malignant_paths_sorted

    # if f=1.0 we dont need to remove anything
    if toremove != 0:
        del benign_paths_small[toremove:-toremove]
        del malignant_paths_small[toremove:-toremove]

    # get the full pathnames
    benign_paths_full = [os.path.join(config['trainingpath'], path) for path in benign_paths_small]
    malignant_paths_full = [os.path.join(config['trainingpath'], path) for path in malignant_paths_small]

    # now save all these images in a new directory
    for imagepath in itertools.chain(benign_paths_full, malignant_paths_full):
        # load image
        image = cv2.imread(imagepath)

        # create a new path to save modified image in
        newpath = imagepath.replace(datasetpath, '{}_{}_f={}'.format(datasetpath, args['modification'], args['fraction']))

        # save image in the new path
        print("Writing image {} ...".format(newpath))
        cv2.imwrite(newpath, image)

if args['modification'] == 'small_clusters':
    # if features are not yet extracted, extract them
    if not os.path.exists(os.path.join(config['output_path'], 'features_cnn.p')):
        # load the pre-trained source network
        print("loading network...")
        dataset = args['dataset']
        modelpath = os.path.join(config['model_savepath'], '{}_model.h5'.format(dataset))
        model = load_model(modelpath)
        model.summary()

        # create network instance
        network = NeuralNetwork(model, config, batchsize=1, seed=seed)

        # set create a bottleneck model at specified layer
        network.set_bottleneck_model(outputlayer='flatten_1')
        network.model.summary()

        # extract features using bottleneck model
        train_features, test_features, true_labels_train, true_labels_test = network.extract_bottleneck_features()

        # get corresponding image paths
        train_paths = network.gen_training.filenames

        # save features in file
        pickle.dump([np.array(train_paths), np.array(train_features)], open(os.path.join(config['output_path'], 'features_cnn.p'), 'wb'))

    else:
        # load features
        train_paths, train_features = pickle.load(open(os.path.join(config['output_path'], 'features_cnn.p'), 'rb'))

    # create list with the right full paths
    train_paths = [os.path.join(config['trainingpath'], filename) for filename in train_paths]

    # if the pdfs are not calculated, calculate, else load them from file to save time
    if not os.path.exists(os.path.join(config['output_path'], 'pdfs.csv')):

        # reshape the flattened feature vector to the original feature maps shape (x, 7, 7, 512)
        train_features = train_features.reshape(train_features.shape[0], 7, 7, 512)

        print(train_features.shape)

        # set the number of bins
        n_bins = 10

        # create empty list to store the concatenated pdfs for every image
        pdfs_images = []

        print(train_features.shape[0])
        # calculate the pdf for every image
        for im_index in range(train_features.shape[0]):
            print("Calculating pdf for image {}".format(train_paths[im_index]))
            print("index {}".format(im_index))
            start = time.time()
            # save from all the feature channels the index, lower bound and upper bound
            # for now, just use first image
            # im_index = 0
            feat_map = train_features[im_index,:,:,:]
            # feat_map = train_features[im_index,:]

            bounds = []
            pdfs = []

            for channel_id in range(feat_map.shape[-1]):
            # for channel_id in [0,1]:
                # take only one of the feature channels and flatten it
                feat_channel = feat_map[:,:,channel_id].flatten()
                # feat_channel = feat_map
                # print(len(feat_channel))
                # print(feat_channel)

                # find the lower and upper bound for this feature channel
                # lower_bound = np.min(feat_channel)
                # upper_bound = np.max(feat_channel)
                #
                # # save the channel index along with the lower and upper bounds in a list
                # bounds.append([channel_id, lower_bound, upper_bound])
                # print(bounds)

                # get the histogram edges when each bin should contain an equal amount of pixels
                # histedges_equal = np.interp(np.linspace(0, len(feat_channel), n_bins+1), np.arange(len(feat_channel)), np.sort(feat_channel))
                # print(histedges_equal)

                # ignore the possible division by zero warning
                # if we want every bin to have roughly equal number of pixels, sometimes
                # a bin has only 0's, and width of this bin will be 0
                # with np.errstate(divide='ignore',invalid='ignore'):
                # create a histogram of the feat_channel
                # density=True makes it a density function, because we want the distribution
                # n, bins, patches = plt.hist(feat_channel, histedges_equal, density=True, edgecolor='black', linewidth=1.2)

                # f = plt.figure()

                # n, bins, patches = plt.hist(feat_channel, density=True, edgecolor='black', linewidth=1.2, histtype='step')
                #
                # print(n)
                # print(bins)


                n, bins = np.histogram(feat_channel, density=True)

                # add the pdf to a list
                pdfs.append(n * np.diff(bins))
                # pdfs.append(pdf)



            # convert the pdfs to an array and concatenate them
            pdfs = np.concatenate(np.asarray(pdfs))

            # add the concatenated pdfs to the list of all pdfs for every image
            pdfs_images.append(pdfs)

        pdfs_images = np.array(pdfs_images)

        # save np array to csv file
        np.savetxt(os.path.join(config['output_path'], 'pdfs.csv'), pdfs_images, delimiter=",")

    else:
        print("loading probability distributions...")
        pdfs_images = np.genfromtxt(os.path.join(config['output_path'], 'pdfs.csv'), delimiter=',')

    # add a small smoothing factor to avoid dealing with zero values and KL values of inf
    pdfs_images = pdfs_images +  0.00001

    print(pdfs_images.shape)

    # the filenames from the generator are for both classes
    # we want to split them up, along with the pdfs values to keep the class balance
    benign_paths, malignant_paths, benign_pdfs, malignant_pdfs = [], [], [], []
    for i, filename in enumerate(train_paths):
        if args['dataset'] in ['ISIC_2', 'ISIC_3', 'ISIC_4', 'ISIC_5', 'ISIC_6']:
            if "malignant" in filename:
                malignant_paths.append(filename)
                malignant_pdfs.append(pdfs_images[i,:])
            if "benign" in filename:
                benign_paths.append(filename)
                benign_pdfs.append(pdfs_images[i,:])
        if args['dataset'] in ['CNMC_2', 'CNMC_3', 'CNMC_4', 'CNMC_5', 'CNMC_6']:
            if "leukemic" in filename:
                malignant_paths.append(filename)
                malignant_pdfs.append(pdfs_images[i,:])
            if "normal" in filename:
                benign_paths.append(filename)
                benign_pdfs.append(pdfs_images[i,:])

    malignant_pdfs = np.array(malignant_pdfs)
    benign_pdfs = np.array(benign_pdfs)

    # calculate the KL divergence
    KL_divergence = lambda p, q : np.sum(p * np.log(p / q))

    for i in range(2):
        if i == 0:
            pdfs = malignant_pdfs
            pdfs_paths = malignant_paths
        if i == 1:
            pdfs = benign_pdfs
            pdfs_paths = benign_paths

        # define the amount of images in a cluster (app. 10%)
        nn = int(0.1*len(pdfs_paths))

        # how many clusters need to be removed
        clusters = int(10 - args['fraction']/0.1)

        print("Clusters to remove: {}".format(clusters))

        # now do for every clusters
        for i in range(clusters):
            # take a random index for an image to find nearest neighbors to
            random_id = np.random.randint(pdfs.shape[0])

            # take the image pdf belonging to the random id
            random_pdf = pdfs[random_id,:]

            KLs = []
            # now calculate the KL divergence of the random pdf with all the other pdfs
            print("Calculating distance between image {} and all other images".format(random_id))
            for i in range(pdfs.shape[0]):
                # calculate the distance between the distributions, two times because KL divergence is not symmetric
                distance = KL_divergence(random_pdf, pdfs[i,:]) + KL_divergence(pdfs[i,:], random_pdf)
                KLs.append(distance)

            # sort the distances and the image paths in the same way
            distances_paths = sorted(zip(KLs, pdfs_paths))

            # create a list of only the sorted image paths
            pdfs_paths_sorted = [pdfs_path for dist,pdfs_path in distances_paths]

            pdfs_paths = pdfs_paths[nn:]
            pdfs = pdfs[nn:,:]

            print("amount of images left: {}".format(len(pdfs_paths)))

        # now save all the images in the remaining paths
        # get the full pathnames
        pdfs_paths_full = [os.path.join(config['trainingpath'], pdf_path) for pdf_path in pdfs_paths]

        # now save all these images in a new directory
        for imagepath in pdfs_paths_full:
            # load image
            image = cv2.imread(imagepath)

            # create a new path to save modified image in
            newpath = imagepath.replace(datasetpath, '{}_{}_f={}'.format(datasetpath, args['modification'], args['fraction']))

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

if args['modification'] == 'image_rot':
    # rotates the images with random amounts of rotation
    for imagepath in mod_paths:
        print(imagepath)
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

if args['modification'] == 'image_translation':
    # translates images in x- and y-direction
    for imagepath in mod_paths:
        # load image
        image = cv2.imread(imagepath)

        # create a new path to save modified image in
        newpath = imagepath.replace(datasetpath, '{}_{}_f={}'.format(datasetpath, args['modification'], args['fraction']))

        # get height and width values
        (h, w) = image.shape[:2]

        # make translation matrix, with 0.1 percent translation
        trans_pct = 0.1
        x_trans = trans_pct * w
        y_trans = trans_pct * h
        mat = np.float32([[1,0,x_trans], [0,1,y_trans]])

        # translate the image
        translated_image = cv2.warpAffine(image, mat, (h, w))

        # save image in the new path
        print("Writing image {} ...".format(newpath))
        cv2.imwrite(newpath, translated_image)

if args['modification'] == 'image_zoom':
    # zooms images
    for imagepath in mod_paths:
        # load image
        image = cv2.imread(imagepath)

        # create a new path to save modified image in
        newpath = imagepath.replace(datasetpath, '{}_{}_f={}'.format(datasetpath, args['modification'], args['fraction']))

        # get height and width values
        (h, w) = image.shape[:2]

        # zoom percentage
        zoom_pct = 1.5

        # calculate offsets
        xmin = int((zoom_pct*w/2)-(w/2))
        xmax = int((w/2)+(zoom_pct*w/2))
        ymin = int((zoom_pct*h/2)-(h/2))
        ymax = int((h/2)+(zoom_pct*h/2))

        # zoom images
        zoomed_image = cv2.resize(image,None,fx=zoom_pct, fy=zoom_pct, interpolation=cv2.INTER_LINEAR)
        zoomed_image = zoomed_image[xmin:xmax,ymin:ymax]

        # save image in the new path
        print("Writing image {} ...".format(newpath))
        cv2.imwrite(newpath, zoomed_image)

if args['modification'] == 'add_noise':
    # adds noise to images
    for imagepath in mod_paths:
        # load image
        image = cv2.imread(imagepath)

        # create a new path to save modified image in
        newpath = imagepath.replace(datasetpath, '{}_{}_{}_f={}'.format(datasetpath, args['modification'], args['noise'], args['fraction']))

        # add gaussian noise
        if args['noise'] == 'gaussian':
            row, col, ch = image.shape
            mean = 0
            var = 0.5
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch)).astype('uint8')
            noisy_image = cv2.add(image, gauss)

        # add poisson noise
        if args['noise'] == 'poisson':
            noise = (np.random.poisson(image / 255.0 * 0.1) / 0.1 * 255).astype('uint8')
            noisy_image = cv2.add(image, noise)

        # add salt and pepper noise
        if args['noise'] == 'salt_and_pepper':
            row, col, ch = image.shape
            salt_vs_pepper = 0.5
            amount = 0.005
            noisy_image = np.copy(image)
            # salt
            num_salt = np.ceil(amount * image.size * salt_vs_pepper)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            noisy_image[tuple(coords)] = 255

            # pepper
            num_pepper = np.ceil(amount* image.size * (1. - salt_vs_pepper))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            noisy_image[tuple(coords)] = 0

        # add speckle noise
        if args['noise'] == 'speckle':
            row, col, ch = image.shape
            gauss = np.random.randn(row, col, ch).astype('uint8')
            gauss = gauss.reshape(row, col, ch)
            noise = image * gauss
            noisy_image = cv2.add(image, gauss)

        # save image in the new path
        print("Writing image {} ...".format(newpath))
        cv2.imwrite(newpath, noisy_image)

if args['modification'] == 'grayscale':
    for imagepath in mod_paths:
        # load image
        image = cv2.imread(imagepath)

        # create a new path to save modified image in
        newpath = imagepath.replace(datasetpath, '{}_{}_f={}'.format(datasetpath, args['modification'], args['fraction']))

        # modify the image to grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # save image in the new path
        print("Writing image {} ...".format(newpath))
        cv2.imwrite(newpath, grayscale_image)

if args['modification'] == 'hsv':
    for imagepath in mod_paths:
        # load image
        image = cv2.imread(imagepath)

        # create a new path to save modified image in
        newpath = imagepath.replace(datasetpath, '{}_{}_f={}'.format(datasetpath, args['modification'], args['fraction']))

        # convert image to HSV values
        HSV_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

        # multiple hue value with random number between 0.8 and 1.2,
        # multiple saturation and brightness with random number between 0.5 and 1.5
        HSV_image = HSV_image.astype('float32')
        HSV_image[:,:,0] *= np.random.uniform(0.8,1.2)
        HSV_image[:,:,1] *= np.random.uniform(0.5,1.5)
        HSV_image[:,:,2] *= np.random.uniform(0.5,1.5)

        # sometimes values get bigger than 255, uint8 cant hold that,
        # so make these values equal to 255 to avoid these values getting low values
        # (and thus black colors instead of white) when converting the image back
        # for hue values greater than 360, make them continue at 0
        HSV_image[:,:,0][HSV_image[:,:,0] > 360] -= 360
        HSV_image[:,:,1][HSV_image[:,:,1] > 255] = 255
        HSV_image[:,:,2][HSV_image[:,:,2] > 255] = 255

        # convert image back to uint8
        HSV_image = HSV_image.astype('uint8')

        # convert image back to BGR colors
        modified_image = cv2.cvtColor(HSV_image,cv2.COLOR_HSV2BGR)

        # save image in the new path
        print("Writing image {} ...".format(newpath))
        cv2.imwrite(newpath, modified_image)

if args['modification'] == 'blur':
    for imagepath in mod_paths:
        # load image
        image = cv2.imread(imagepath)

        # create a new path to save modified image in
        newpath = imagepath.replace(datasetpath, '{}_{}_f={}'.format(datasetpath, args['modification'], args['fraction']))

        # modify the image to grayscale
        blurred_image = cv2.GaussianBlur(image,(3,3),3.0)

        # save image in the new path
        print("Writing image {} ...".format(newpath))
        cv2.imwrite(newpath, blurred_image)


if args['modification'] == 'image_rot' or args['modification'] == 'image_translation' or args['modification'] == 'image_zoom' or args['modification'] == 'add_noise' or args['modification'] == 'grayscale' or args['modification'] == 'hsv' or args['modification'] == 'blur':
    # save the rest of the training images
    for imagepath in rest_paths:
        # do the same, but without modification
        image = cv2.imread(imagepath)

        if args['modification'] == 'add_noise':
            newpath = imagepath.replace(datasetpath, '{}_{}_{}_f={}'.format(datasetpath, args['modification'], args['noise'], args['fraction']))
        else:
            newpath = imagepath.replace(datasetpath, '{}_{}_f={}'.format(datasetpath, args['modification'], args['fraction']))

        print("Writing image {} ...".format(newpath))
        cv2.imwrite(newpath, image)

# always we need to just copy the validation and test set into the new dataset directory, except for when changing the split
if args['modification'] != 'change_split':
    for imagepath in itertools.chain(validationpaths, testpaths):
        # load image
        image = cv2.imread(imagepath)

        # create a new path to save modified image in
        if args['modification'] == 'add_noise':
            newpath = imagepath.replace(datasetpath, '{}_{}_{}_f={}'.format(datasetpath, args['modification'], args['noise'], args['fraction']))

        else:
            newpath = imagepath.replace(datasetpath, '{}_{}_f={}'.format(datasetpath, args['modification'], args['fraction']))

        # save image in the new path
        print("Writing image {} ...".format(newpath))
        cv2.imwrite(newpath, image)
