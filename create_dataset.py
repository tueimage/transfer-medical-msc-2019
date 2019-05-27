# script to build dataset in a generic structure
# structure should be dataset_name/split/class/images.ext

import config_ISIC
import os
import glob
from shutil import copy2

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

            # 0 for nevus or seborrheic keratosis
            if int(float(label1)) == 0:
                # copy images belonging to class 'nevus or seborrheic keratosis'
                impath = os.path.join(imdir, '{}.jpg'.format(im))
                copy2(impath, target_impath_0)

            # 1 for melanoma
            elif int(float(label1)) == 1:
                # copy images belonging to class 'melanoma'
                impath = os.path.join(imdir, '{}.jpg'.format(im))
                copy2(impath, target_impath_1)
