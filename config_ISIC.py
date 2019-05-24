# code that sorts dataset and saves it in a generic format
# format should be dataset_folder/split/class/image.ext

import os

# path to original dataset
PARENT_PATH = os.path.dirname(os.getcwd())
PATH_ORIG_DATA = os.path.join(PARENT_PATH, "Data/ISIC-2017/ISIC-2017_Training_Data")

# base path after splitting data
BASE_PATH = os.path.join(PARENT_PATH, "dataset/dataset_ISIC")

# split datapaths
TRAIN = "training"
VAL = "validation"
TEST = "testing"

# initialize class label names, melanoma vs nevus or seborrheic keratosis
CLASSES = ["melanoma", "nevus_sk"]

# set batch size
BATCH_SIZE = 8

# initialize label encoder path and output directory for extracted features
LE_PATH = os.path.join(PARENT_PATH, "outputs/output_ISIC", "le.cpickle")
BASE_CSV_PATH = os.path.join(PARENT_PATH, "output/output_ISIC")

# path for serialized model after training
MODEL_PATH = os.path.join(PARENT_PATH, "output/output_ISIC", "model.cpickle")
