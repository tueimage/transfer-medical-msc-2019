import os

# path to original dataset, use parent path so data is not in same repo as code
PARENT_PATH = os.path.dirname(os.getcwd())
PATH_ORIG_DATA = os.path.join(PARENT_PATH, "Data/ISIC-2017")

# base path after splitting data
DATASET_PATH = os.path.join(PARENT_PATH, "datasets/dataset_ISIC")

# split datapaths
TRAIN = "training"
VAL = "validation"
TEST = "test"

# initialize class label names, melanoma vs nevus or seborrheic keratosis
CLASSES = ["melanoma", "nevus_sk"]

# set batch size
BATCH_SIZE = 8

# initialize label encoder path and output directory for extracted features
LE_PATH = os.path.join(PARENT_PATH, "outputs/output_ISIC", "le.cpickle")
BASE_CSV_PATH = os.path.join(PARENT_PATH, "outputs/output_ISIC")

# path for serialized model after training
MODEL_PATH = os.path.join(PARENT_PATH, "outputs/output_ISIC", "model.cpickle")

# path to save training plots for fine-tuning
PLOT_PATH = os.path.join(PARENT_PATH, "outputs/output_ISIC/plots")
