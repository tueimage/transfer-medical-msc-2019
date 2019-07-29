# script to build dataset in a generic structure
# structure should be dataset_name/split/class/images.ext

import config_ISIC
import os
import glob
import cv2
import numpy as np

for split in (config_ISIC.TRAIN, config_ISIC.VAL, config_ISIC.TEST):
    print("processing '{} split'...".format(split))

    # create path where images in corresponding split should be saved
    savedir = os.path.join(config_ISIC.DATASET_PATH, split)

    # find path to images corresponding to split
    imdir = glob.glob(os.path.join(config_ISIC.PATH_ORIG_DATA, '*{}*Data'.format(split)))[0]

    # find path to ground truth csv file
    gt_path = glob.glob(os.path.join(config_ISIC.PATH_ORIG_DATA, '*{}*.csv'.format(split)))[0]

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

            if int(float(label1)) == 0:
                # save images which belong to nevus or seborrheic keratosis class (label 0)
                cv2.imwrite(os.path.join(target_impath_0, '{}.jpg'.format(im)), resized_image)

            elif int(float(label1)) == 1:
                # save images which belong to nevus or seborrheic keratosis class (label 0)
                cv2.imwrite(os.path.join(target_impath_1, '{}.jpg'.format(im)), resized_image)
