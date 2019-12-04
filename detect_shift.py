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
import matplotlib.pyplot as plt
from models import get_MLP
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from utils import *
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from scipy.stats import binom_test, kstest, ks_2samp
from openpyxl import load_workbook, Workbook

# choose GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# construct argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m',
    '--mode',
    choices=['train_val', 'datasets'],
    required=True,
    help='dataset shift to detect, train_val for shift between train and validation set')
parser.add_argument('-d',
    '--dataset',
    choices=['ISIC_2', 'ISIC_3', 'ISIC_4', 'ISIC_5', 'ISIC_6'],
    required=True,
    help='dataset to use')
parser.add_argument('-d2',
    '--dataset2',
    choices=['ISIC_2', 'ISIC_2_image_rot_f=0.1', 'ISIC_2_image_rot_f=0.2',
            'ISIC_2_image_rot_f=0.3', 'ISIC_2_image_rot_f=0.4', 'ISIC_2_image_rot_f=0.5',
            'ISIC_2_image_rot_f=0.6', 'ISIC_2_image_rot_f=0.7', 'ISIC_2_image_rot_f=0.8',
            'ISIC_2_image_rot_f=0.9', 'ISIC_2_image_rot_f=1.0', 'ISIC_2_image_translation_f=0.1',
            'ISIC_2_image_translation_f=0.2', 'ISIC_2_image_translation_f=0.3', 'ISIC_2_image_translation_f=0.4',
            'ISIC_2_image_translation_f=0.5', 'ISIC_2_image_translation_f=0.6', 'ISIC_2_image_translation_f=0.7',
            'ISIC_2_image_translation_f=0.8', 'ISIC_2_image_translation_f=0.9', 'ISIC_2_image_translation_f=1.0',
            'ISIC_2_image_zoom_f=0.1', 'ISIC_2_image_zoom_f=0.2', 'ISIC_2_image_zoom_f=0.3',
            'ISIC_2_image_zoom_f=0.4', 'ISIC_2_image_zoom_f=0.5', 'ISIC_2_image_zoom_f=0.6',
            'ISIC_2_image_zoom_f=0.7', 'ISIC_2_image_zoom_f=0.8', 'ISIC_2_image_zoom_f=0.9',
            'ISIC_2_image_zoom_f=1.0', 'ISIC_2_add_noise_gaussian_f=0.1', 'ISIC_2_add_noise_gaussian_f=0.2',
            'ISIC_2_add_noise_gaussian_f=0.3', 'ISIC_2_add_noise_gaussian_f=0.4', 'ISIC_2_add_noise_gaussian_f=0.5',
            'ISIC_2_add_noise_gaussian_f=0.6', 'ISIC_2_add_noise_gaussian_f=0.7', 'ISIC_2_add_noise_gaussian_f=0.8',
            'ISIC_2_add_noise_gaussian_f=0.9', 'ISIC_2_add_noise_gaussian_f=1.0', 'ISIC_2_add_noise_poisson_f=0.1',
            'ISIC_2_add_noise_poisson_f=0.2', 'ISIC_2_add_noise_poisson_f=0.3', 'ISIC_2_add_noise_poisson_f=0.4',
            'ISIC_2_add_noise_poisson_f=0.5', 'ISIC_2_add_noise_poisson_f=0.6', 'ISIC_2_add_noise_poisson_f=0.7',
            'ISIC_2_add_noise_poisson_f=0.8', 'ISIC_2_add_noise_poisson_f=0.9', 'ISIC_2_add_noise_poisson_f=1.0',
            'ISIC_2_add_noise_salt_and_pepper_f=0.1', 'ISIC_2_add_noise_salt_and_pepper_f=0.2',
            'ISIC_2_add_noise_salt_and_pepper_f=0.3', 'ISIC_2_add_noise_salt_and_pepper_f=0.4',
            'ISIC_2_add_noise_salt_and_pepper_f=0.5', 'ISIC_2_add_noise_salt_and_pepper_f=0.6',
            'ISIC_2_add_noise_salt_and_pepper_f=0.7', 'ISIC_2_add_noise_salt_and_pepper_f=0.8',
            'ISIC_2_add_noise_salt_and_pepper_f=0.9', 'ISIC_2_add_noise_salt_and_pepper_f=1.0',
            'ISIC_2_add_noise_speckle_f=0.1', 'ISIC_2_add_noise_speckle_f=0.2', 'ISIC_2_add_noise_speckle_f=0.3',
            'ISIC_2_add_noise_speckle_f=0.4', 'ISIC_2_add_noise_speckle_f=0.5', 'ISIC_2_add_noise_speckle_f=0.6',
            'ISIC_2_add_noise_speckle_f=0.7', 'ISIC_2_add_noise_speckle_f=0.8', 'ISIC_2_add_noise_speckle_f=0.9',
            'ISIC_2_add_noise_speckle_f=1.0', 'ISIC_2_imbalance_classes_f=0.1', 'ISIC_2_imbalance_classes_f=0.2',
            'ISIC_2_imbalance_classes_f=0.3', 'ISIC_2_imbalance_classes_f=0.4', 'ISIC_2_imbalance_classes_f=0.5',
            'ISIC_2_imbalance_classes_f=0.6', 'ISIC_2_imbalance_classes_f=0.7', 'ISIC_2_imbalance_classes_f=0.8',
            'ISIC_2_imbalance_classes_f=0.9', 'ISIC_2_imbalance_classes_f=1.0',
            'ISIC_3', 'ISIC_3_image_rot_f=0.1', 'ISIC_3_image_rot_f=0.2',
            'ISIC_3_image_rot_f=0.3', 'ISIC_3_image_rot_f=0.4', 'ISIC_3_image_rot_f=0.5',
            'ISIC_3_image_rot_f=0.6', 'ISIC_3_image_rot_f=0.7', 'ISIC_3_image_rot_f=0.8',
            'ISIC_3_image_rot_f=0.9', 'ISIC_3_image_rot_f=1.0', 'ISIC_3_image_translation_f=0.1',
            'ISIC_3_image_translation_f=0.2', 'ISIC_3_image_translation_f=0.3', 'ISIC_3_image_translation_f=0.4',
            'ISIC_3_image_translation_f=0.5', 'ISIC_3_image_translation_f=0.6', 'ISIC_3_image_translation_f=0.7',
            'ISIC_3_image_translation_f=0.8', 'ISIC_3_image_translation_f=0.9', 'ISIC_3_image_translation_f=1.0',
            'ISIC_3_image_zoom_f=0.1', 'ISIC_3_image_zoom_f=0.2', 'ISIC_3_image_zoom_f=0.3',
            'ISIC_3_image_zoom_f=0.4', 'ISIC_3_image_zoom_f=0.5', 'ISIC_3_image_zoom_f=0.6',
            'ISIC_3_image_zoom_f=0.7', 'ISIC_3_image_zoom_f=0.8', 'ISIC_3_image_zoom_f=0.9',
            'ISIC_3_image_zoom_f=1.0', 'ISIC_3_add_noise_gaussian_f=0.1', 'ISIC_3_add_noise_gaussian_f=0.2',
            'ISIC_3_add_noise_gaussian_f=0.3', 'ISIC_3_add_noise_gaussian_f=0.4', 'ISIC_3_add_noise_gaussian_f=0.5',
            'ISIC_3_add_noise_gaussian_f=0.6', 'ISIC_3_add_noise_gaussian_f=0.7', 'ISIC_3_add_noise_gaussian_f=0.8',
            'ISIC_3_add_noise_gaussian_f=0.9', 'ISIC_3_add_noise_gaussian_f=1.0', 'ISIC_3_add_noise_poisson_f=0.1',
            'ISIC_3_add_noise_poisson_f=0.2', 'ISIC_3_add_noise_poisson_f=0.3', 'ISIC_3_add_noise_poisson_f=0.4',
            'ISIC_3_add_noise_poisson_f=0.5', 'ISIC_3_add_noise_poisson_f=0.6', 'ISIC_3_add_noise_poisson_f=0.7',
            'ISIC_3_add_noise_poisson_f=0.8', 'ISIC_3_add_noise_poisson_f=0.9', 'ISIC_3_add_noise_poisson_f=1.0',
            'ISIC_3_add_noise_salt_and_pepper_f=0.1', 'ISIC_3_add_noise_salt_and_pepper_f=0.2',
            'ISIC_3_add_noise_salt_and_pepper_f=0.3', 'ISIC_3_add_noise_salt_and_pepper_f=0.4',
            'ISIC_3_add_noise_salt_and_pepper_f=0.5', 'ISIC_3_add_noise_salt_and_pepper_f=0.6',
            'ISIC_3_add_noise_salt_and_pepper_f=0.7', 'ISIC_3_add_noise_salt_and_pepper_f=0.8',
            'ISIC_3_add_noise_salt_and_pepper_f=0.9', 'ISIC_3_add_noise_salt_and_pepper_f=1.0',
            'ISIC_3_add_noise_speckle_f=0.1', 'ISIC_3_add_noise_speckle_f=0.2', 'ISIC_3_add_noise_speckle_f=0.3',
            'ISIC_3_add_noise_speckle_f=0.4', 'ISIC_3_add_noise_speckle_f=0.5', 'ISIC_3_add_noise_speckle_f=0.6',
            'ISIC_3_add_noise_speckle_f=0.7', 'ISIC_3_add_noise_speckle_f=0.8', 'ISIC_3_add_noise_speckle_f=0.9',
            'ISIC_3_add_noise_speckle_f=1.0', 'ISIC_3_imbalance_classes_f=0.1', 'ISIC_3_imbalance_classes_f=0.2',
            'ISIC_3_imbalance_classes_f=0.3', 'ISIC_3_imbalance_classes_f=0.4', 'ISIC_3_imbalance_classes_f=0.5',
            'ISIC_3_imbalance_classes_f=0.6', 'ISIC_3_imbalance_classes_f=0.7', 'ISIC_3_imbalance_classes_f=0.8',
            'ISIC_3_imbalance_classes_f=0.9', 'ISIC_3_imbalance_classes_f=1.0',
            'ISIC_4', 'ISIC_4_image_rot_f=0.1', 'ISIC_4_image_rot_f=0.2',
            'ISIC_4_image_rot_f=0.3', 'ISIC_4_image_rot_f=0.4', 'ISIC_4_image_rot_f=0.5',
            'ISIC_4_image_rot_f=0.6', 'ISIC_4_image_rot_f=0.7', 'ISIC_4_image_rot_f=0.8',
            'ISIC_4_image_rot_f=0.9', 'ISIC_4_image_rot_f=1.0', 'ISIC_4_image_translation_f=0.1',
            'ISIC_4_image_translation_f=0.2', 'ISIC_4_image_translation_f=0.3', 'ISIC_4_image_translation_f=0.4',
            'ISIC_4_image_translation_f=0.5', 'ISIC_4_image_translation_f=0.6', 'ISIC_4_image_translation_f=0.7',
            'ISIC_4_image_translation_f=0.8', 'ISIC_4_image_translation_f=0.9', 'ISIC_4_image_translation_f=1.0',
            'ISIC_4_image_zoom_f=0.1', 'ISIC_4_image_zoom_f=0.2', 'ISIC_4_image_zoom_f=0.3',
            'ISIC_4_image_zoom_f=0.4', 'ISIC_4_image_zoom_f=0.5', 'ISIC_4_image_zoom_f=0.6',
            'ISIC_4_image_zoom_f=0.7', 'ISIC_4_image_zoom_f=0.8', 'ISIC_4_image_zoom_f=0.9',
            'ISIC_4_image_zoom_f=1.0', 'ISIC_4_add_noise_gaussian_f=0.1', 'ISIC_4_add_noise_gaussian_f=0.2',
            'ISIC_4_add_noise_gaussian_f=0.3', 'ISIC_4_add_noise_gaussian_f=0.4', 'ISIC_4_add_noise_gaussian_f=0.5',
            'ISIC_4_add_noise_gaussian_f=0.6', 'ISIC_4_add_noise_gaussian_f=0.7', 'ISIC_4_add_noise_gaussian_f=0.8',
            'ISIC_4_add_noise_gaussian_f=0.9', 'ISIC_4_add_noise_gaussian_f=1.0', 'ISIC_4_add_noise_poisson_f=0.1',
            'ISIC_4_add_noise_poisson_f=0.2', 'ISIC_4_add_noise_poisson_f=0.3', 'ISIC_4_add_noise_poisson_f=0.4',
            'ISIC_4_add_noise_poisson_f=0.5', 'ISIC_4_add_noise_poisson_f=0.6', 'ISIC_4_add_noise_poisson_f=0.7',
            'ISIC_4_add_noise_poisson_f=0.8', 'ISIC_4_add_noise_poisson_f=0.9', 'ISIC_4_add_noise_poisson_f=1.0',
            'ISIC_4_add_noise_salt_and_pepper_f=0.1', 'ISIC_4_add_noise_salt_and_pepper_f=0.2',
            'ISIC_4_add_noise_salt_and_pepper_f=0.3', 'ISIC_4_add_noise_salt_and_pepper_f=0.4',
            'ISIC_4_add_noise_salt_and_pepper_f=0.5', 'ISIC_4_add_noise_salt_and_pepper_f=0.6',
            'ISIC_4_add_noise_salt_and_pepper_f=0.7', 'ISIC_4_add_noise_salt_and_pepper_f=0.8',
            'ISIC_4_add_noise_salt_and_pepper_f=0.9', 'ISIC_4_add_noise_salt_and_pepper_f=1.0',
            'ISIC_4_add_noise_speckle_f=0.1', 'ISIC_4_add_noise_speckle_f=0.2', 'ISIC_4_add_noise_speckle_f=0.3',
            'ISIC_4_add_noise_speckle_f=0.4', 'ISIC_4_add_noise_speckle_f=0.5', 'ISIC_4_add_noise_speckle_f=0.6',
            'ISIC_4_add_noise_speckle_f=0.7', 'ISIC_4_add_noise_speckle_f=0.8', 'ISIC_4_add_noise_speckle_f=0.9',
            'ISIC_4_add_noise_speckle_f=1.0', 'ISIC_4_imbalance_classes_f=0.1', 'ISIC_4_imbalance_classes_f=0.2',
            'ISIC_4_imbalance_classes_f=0.3', 'ISIC_4_imbalance_classes_f=0.4', 'ISIC_4_imbalance_classes_f=0.5',
            'ISIC_4_imbalance_classes_f=0.6', 'ISIC_4_imbalance_classes_f=0.7', 'ISIC_4_imbalance_classes_f=0.8',
            'ISIC_4_imbalance_classes_f=0.9', 'ISIC_4_imbalance_classes_f=1.0',
            'ISIC_5', 'ISIC_5_image_rot_f=0.1', 'ISIC_5_image_rot_f=0.2',
            'ISIC_5_image_rot_f=0.3', 'ISIC_5_image_rot_f=0.4', 'ISIC_5_image_rot_f=0.5',
            'ISIC_5_image_rot_f=0.6', 'ISIC_5_image_rot_f=0.7', 'ISIC_5_image_rot_f=0.8',
            'ISIC_5_image_rot_f=0.9', 'ISIC_5_image_rot_f=1.0', 'ISIC_5_image_translation_f=0.1',
            'ISIC_5_image_translation_f=0.2', 'ISIC_5_image_translation_f=0.3', 'ISIC_5_image_translation_f=0.4',
            'ISIC_5_image_translation_f=0.5', 'ISIC_5_image_translation_f=0.6', 'ISIC_5_image_translation_f=0.7',
            'ISIC_5_image_translation_f=0.8', 'ISIC_5_image_translation_f=0.9', 'ISIC_5_image_translation_f=1.0',
            'ISIC_5_image_zoom_f=0.1', 'ISIC_5_image_zoom_f=0.2', 'ISIC_5_image_zoom_f=0.3',
            'ISIC_5_image_zoom_f=0.4', 'ISIC_5_image_zoom_f=0.5', 'ISIC_5_image_zoom_f=0.6',
            'ISIC_5_image_zoom_f=0.7', 'ISIC_5_image_zoom_f=0.8', 'ISIC_5_image_zoom_f=0.9',
            'ISIC_5_image_zoom_f=1.0', 'ISIC_5_add_noise_gaussian_f=0.1', 'ISIC_5_add_noise_gaussian_f=0.2',
            'ISIC_5_add_noise_gaussian_f=0.3', 'ISIC_5_add_noise_gaussian_f=0.4', 'ISIC_5_add_noise_gaussian_f=0.5',
            'ISIC_5_add_noise_gaussian_f=0.6', 'ISIC_5_add_noise_gaussian_f=0.7', 'ISIC_5_add_noise_gaussian_f=0.8',
            'ISIC_5_add_noise_gaussian_f=0.9', 'ISIC_5_add_noise_gaussian_f=1.0', 'ISIC_5_add_noise_poisson_f=0.1',
            'ISIC_5_add_noise_poisson_f=0.2', 'ISIC_5_add_noise_poisson_f=0.3', 'ISIC_5_add_noise_poisson_f=0.4',
            'ISIC_5_add_noise_poisson_f=0.5', 'ISIC_5_add_noise_poisson_f=0.6', 'ISIC_5_add_noise_poisson_f=0.7',
            'ISIC_5_add_noise_poisson_f=0.8', 'ISIC_5_add_noise_poisson_f=0.9', 'ISIC_5_add_noise_poisson_f=1.0',
            'ISIC_5_add_noise_salt_and_pepper_f=0.1', 'ISIC_5_add_noise_salt_and_pepper_f=0.2',
            'ISIC_5_add_noise_salt_and_pepper_f=0.3', 'ISIC_5_add_noise_salt_and_pepper_f=0.4',
            'ISIC_5_add_noise_salt_and_pepper_f=0.5', 'ISIC_5_add_noise_salt_and_pepper_f=0.6',
            'ISIC_5_add_noise_salt_and_pepper_f=0.7', 'ISIC_5_add_noise_salt_and_pepper_f=0.8',
            'ISIC_5_add_noise_salt_and_pepper_f=0.9', 'ISIC_5_add_noise_salt_and_pepper_f=1.0',
            'ISIC_5_add_noise_speckle_f=0.1', 'ISIC_5_add_noise_speckle_f=0.2', 'ISIC_5_add_noise_speckle_f=0.3',
            'ISIC_5_add_noise_speckle_f=0.4', 'ISIC_5_add_noise_speckle_f=0.5', 'ISIC_5_add_noise_speckle_f=0.6',
            'ISIC_5_add_noise_speckle_f=0.7', 'ISIC_5_add_noise_speckle_f=0.8', 'ISIC_5_add_noise_speckle_f=0.9',
            'ISIC_5_add_noise_speckle_f=1.0', 'ISIC_5_imbalance_classes_f=0.1', 'ISIC_5_imbalance_classes_f=0.2',
            'ISIC_5_imbalance_classes_f=0.3', 'ISIC_5_imbalance_classes_f=0.4', 'ISIC_5_imbalance_classes_f=0.5',
            'ISIC_5_imbalance_classes_f=0.6', 'ISIC_5_imbalance_classes_f=0.7', 'ISIC_5_imbalance_classes_f=0.8',
            'ISIC_5_imbalance_classes_f=0.9', 'ISIC_5_imbalance_classes_f=1.0',
            'ISIC_6', 'ISIC_6_image_rot_f=0.1', 'ISIC_6_image_rot_f=0.2',
            'ISIC_6_image_rot_f=0.3', 'ISIC_6_image_rot_f=0.4', 'ISIC_6_image_rot_f=0.5',
            'ISIC_6_image_rot_f=0.6', 'ISIC_6_image_rot_f=0.7', 'ISIC_6_image_rot_f=0.8',
            'ISIC_6_image_rot_f=0.9', 'ISIC_6_image_rot_f=1.0', 'ISIC_6_image_translation_f=0.1',
            'ISIC_6_image_translation_f=0.2', 'ISIC_6_image_translation_f=0.3', 'ISIC_6_image_translation_f=0.4',
            'ISIC_6_image_translation_f=0.5', 'ISIC_6_image_translation_f=0.6', 'ISIC_6_image_translation_f=0.7',
            'ISIC_6_image_translation_f=0.8', 'ISIC_6_image_translation_f=0.9', 'ISIC_6_image_translation_f=1.0',
            'ISIC_6_image_zoom_f=0.1', 'ISIC_6_image_zoom_f=0.2', 'ISIC_6_image_zoom_f=0.3',
            'ISIC_6_image_zoom_f=0.4', 'ISIC_6_image_zoom_f=0.5', 'ISIC_6_image_zoom_f=0.6',
            'ISIC_6_image_zoom_f=0.7', 'ISIC_6_image_zoom_f=0.8', 'ISIC_6_image_zoom_f=0.9',
            'ISIC_6_image_zoom_f=1.0', 'ISIC_6_add_noise_gaussian_f=0.1', 'ISIC_6_add_noise_gaussian_f=0.2',
            'ISIC_6_add_noise_gaussian_f=0.3', 'ISIC_6_add_noise_gaussian_f=0.4', 'ISIC_6_add_noise_gaussian_f=0.5',
            'ISIC_6_add_noise_gaussian_f=0.6', 'ISIC_6_add_noise_gaussian_f=0.7', 'ISIC_6_add_noise_gaussian_f=0.8',
            'ISIC_6_add_noise_gaussian_f=0.9', 'ISIC_6_add_noise_gaussian_f=1.0', 'ISIC_6_add_noise_poisson_f=0.1',
            'ISIC_6_add_noise_poisson_f=0.2', 'ISIC_6_add_noise_poisson_f=0.3', 'ISIC_6_add_noise_poisson_f=0.4',
            'ISIC_6_add_noise_poisson_f=0.5', 'ISIC_6_add_noise_poisson_f=0.6', 'ISIC_6_add_noise_poisson_f=0.7',
            'ISIC_6_add_noise_poisson_f=0.8', 'ISIC_6_add_noise_poisson_f=0.9', 'ISIC_6_add_noise_poisson_f=1.0',
            'ISIC_6_add_noise_salt_and_pepper_f=0.1', 'ISIC_6_add_noise_salt_and_pepper_f=0.2',
            'ISIC_6_add_noise_salt_and_pepper_f=0.3', 'ISIC_6_add_noise_salt_and_pepper_f=0.4',
            'ISIC_6_add_noise_salt_and_pepper_f=0.5', 'ISIC_6_add_noise_salt_and_pepper_f=0.6',
            'ISIC_6_add_noise_salt_and_pepper_f=0.7', 'ISIC_6_add_noise_salt_and_pepper_f=0.8',
            'ISIC_6_add_noise_salt_and_pepper_f=0.9', 'ISIC_6_add_noise_salt_and_pepper_f=1.0',
            'ISIC_6_add_noise_speckle_f=0.1', 'ISIC_6_add_noise_speckle_f=0.2', 'ISIC_6_add_noise_speckle_f=0.3',
            'ISIC_6_add_noise_speckle_f=0.4', 'ISIC_6_add_noise_speckle_f=0.5', 'ISIC_6_add_noise_speckle_f=0.6',
            'ISIC_6_add_noise_speckle_f=0.7', 'ISIC_6_add_noise_speckle_f=0.8', 'ISIC_6_add_noise_speckle_f=0.9',
            'ISIC_6_add_noise_speckle_f=1.0', 'ISIC_6_imbalance_classes_f=0.1', 'ISIC_6_imbalance_classes_f=0.2',
            'ISIC_6_imbalance_classes_f=0.3', 'ISIC_6_imbalance_classes_f=0.4', 'ISIC_6_imbalance_classes_f=0.5',
            'ISIC_6_imbalance_classes_f=0.6', 'ISIC_6_imbalance_classes_f=0.7', 'ISIC_6_imbalance_classes_f=0.8',
            'ISIC_6_imbalance_classes_f=0.9', 'ISIC_6_imbalance_classes_f=1.0'],
    required='datasets' in sys.argv,
    help='second dataset to use')

args = vars(parser.parse_args())

# read parameters for wanted dataset from config file
with open('config.json') as json_file:
    file = json.load(json_file)
    config = file[args['dataset']]
    config2 = file[args['dataset2']]

# create results directory if it doesn't exist
resultspath = os.path.join(os.path.dirname(os.getcwd()), 'results')
if not os.path.exists(resultspath):
    os.makedirs(resultspath)

if args['mode'] == 'train_val':
    # get training and validation path
    trainingpath = config['trainingpath']
    validationpath = config['validationpath']

    print(trainingpath)
    print(validationpath)

    # get image paths
    impaths_train = glob.glob(os.path.join(trainingpath, '**/*.jpg'))
    impaths_val = glob.glob(os.path.join(validationpath, '**/*.jpg'))

    # now split both datasplits in half to use for training and testing
    impaths_train_t = impaths_train[:int(np.ceil(0.5*len(impaths_train)))]
    impaths_val_t = impaths_val[:int(np.ceil(0.5*len(impaths_val)))]

    impaths_train_s = impaths_train[int(np.ceil(0.5*len(impaths_train))):]
    impaths_val_s = impaths_val[int(np.ceil(0.5*len(impaths_val))):]

    print(len(impaths_train_t), len(impaths_train_s))
    print(len(impaths_val_t), len(impaths_val_s))

if args['mode'] == 'datasets':
    trainingpath1 = config['trainingpath']
    trainingpath2 = config2['trainingpath']

    impaths_1 = glob.glob(os.path.join(trainingpath1, '**/*.jpg'))
    impaths_2 = glob.glob(os.path.join(trainingpath2, '**/*.jpg'))

    # now split both datasplits in half to use for training and testing
    impaths_train_t = impaths_1[:int(np.ceil(0.5*len(impaths_1)))]
    impaths_val_t = impaths_2[:int(np.ceil(0.5*len(impaths_2)))]

    impaths_train_s = impaths_1[int(np.ceil(0.5*len(impaths_1))):]
    impaths_val_s = impaths_2[int(np.ceil(0.5*len(impaths_2))):]

    print(len(impaths_train_t), len(impaths_train_s))
    print(len(impaths_val_t), len(impaths_val_s))


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

# flatten images to single feature vector
x_train = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:]))
x_val = x_val.reshape(x_val.shape[0], np.prod(x_val.shape[1:]))

# convert arrays to float32
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')

# scale the data between 0 and 1
x_train /= 255.0
x_val /= 255.0

# one-hot encode the labels
# y_train_one_hot = to_categorical(y_train)
# y_val_one_hot = to_categorical(y_val)

# get MLP model
MLP = get_MLP(input_shape=(x_train.shape[1],))

MLP.summary()

# configure optimizer
sgd = SGD(lr=1e-5, nesterov=True)

# compile model
MLP.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

# train model
hist = MLP.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(x_val, y_val))

# now make predictions
print("evaluating model...")
preds = MLP.predict(x_val, batch_size=1, verbose=1)

print(preds)
print(np.unique(preds))

# preds is an array like [[x] [x] [x]], make it into array like [x x x]
preds = np.asarray([label for sublist in preds for label in sublist])

# get true labels
true_labels = y_val

# calculate AUC and sklearn AUC
fpr, tpr, thresholds, AUC = AUC_score(preds, true_labels)
skfpr, sktpr, skthresholds, skAUC = skAUC_score(preds, true_labels)

# calculate accuracy score
acc = accuracy(preds, true_labels)


pred_labels = np.where(preds > 0.5, 1, 0).astype(int)



unique, counts = np.unique(true_labels, return_counts=True)
print("distribution of true labels: {}".format(dict(zip(unique, counts))))

unique, counts = np.unique(pred_labels, return_counts=True)
print("distribution of predicted labels: {}".format(dict(zip(unique, counts))))


# print(max(ACC_list))

print("accuracy: {}".format(accuracy))

# p_val_list = []
# for i in range(len(true_labels)):
#     p_val = binom_test((i+1), len(true_labels), p=0.5)
#     p_val_list.append(p_val)

print(true_labels)
print(pred_labels)

# get the number of successes
TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))

num_successes = TP + TN

print("successes: {}".format(num_successes))
print("total: {}".format(len(true_labels)))

p_val_test = binom_test(num_successes, len(true_labels), p=0.5)

print("p-value: {}".format(p_val_test))

stat_ks, p_val_ks = ks_2samp(pred_labels, true_labels, alternative='two-sided')
print(stat_ks, p_val_ks)



absAUC = round(abs(skAUC-0.5),3)

# create a savepath for results and create a sheet to avoid errors
savefile = os.path.join(config['output_path'], 'results_shift.xlsx')

# also create excel file already to avoid errors, if it doesn't exist yet
if not os.path.exists(savefile):
    Workbook().save(savefile)

# save in path
test_results = pd.Series({'dataset2': args['dataset2'], 'absAUC': absAUC, 'AUC': AUC, 'skAUC': skAUC, 'acc': acc, 'p_val_binom': p_val_test, 'stat_ks': stat_ks, 'p_val_ks': p_val_ks})

# read existing rows and add test results
try:
    df = pd.read_excel(savefile, sheet_name='detect_shift')
except:
    df = pd.DataFrame(columns=['dataset2', 'absAUC', 'AUC', 'skAUC', 'acc', 'p_val_binom', 'stat_ks', 'p_val_ks'])

df = df.append(test_results, ignore_index=True)
df.set_index('dataset2', inplace=True)

# save results in excel file
with pd.ExcelWriter(savefile, engine='openpyxl') as writer:
    writer.book = load_workbook(savefile)
    writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
    df.index.name = 'dataset2'
    df.to_excel(writer, sheet_name='detect_shift')


# stat_ks, p_val_ks = kstest(pred_labels, N=len(pred_labels), alternative='two-sided')
# print(stat_ks, p_val_ks)

# get path to save csv file for results
# shift_csvpath = os.path.join(resultspath, 'shift_results.csv')
# print(shift_csvpath)
#
# # create csv file if it doesn't exist yet with the correct headers
# if not os.path.exists(shift_csvpath):
#     # initialize dataframe to save results for different combinations of datasets and add to csv file
#     shift_results = pd.DataFrame(columns=['dataset1', 'dataset2', 'AUC', 'skl_AUC', 'ACC', 'p_val_binom', 'stat_ks', 'p_val_ks'])
#     shift_results.to_csv(shift_csvpath, index=False)
#
# # add new results to dataframe
# row = pd.Series({'dataset1': args['dataset'],
#                 'dataset2': args['dataset2'],
#                 'AUC': AUC,
#                 'skl_AUC': AUC2,
#                 'ACC': accuracy,
#                 'p_val_binom': p_val_test,
#                 'stat_ks': stat_ks,
#                 'p_val_ks': p_val_ks})
#
# # read existing dataframe, add new row and save again
# shift_results = pd.read_csv(shift_csvpath)
# shift_results = shift_results.append(row, ignore_index=True)
# shift_results.to_csv(shift_csvpath, index=False)















#
# # shuffle the arrays with same seed
# sd=28
# np.random.seed(sd)
# np.random.shuffle(x_train)
# np.random.shuffle(y_train)
#
# # get model to train classifier
# input_tensor=Input(shape=(224,224,3))
# model_VGG16 = model_VGG16_light(input_tensor)
#
# # initialize image data generator objects
# gen_obj_training = ImageDataGenerator(rescale=1./255, featurewise_center=True)
# gen_obj_test = ImageDataGenerator(rescale=1./255, featurewise_center=True)
#
# # we need to fit generators to training data
# # from this mean and std, featurewise_center is calculated in the generator
# gen_obj_training.fit(x_train, seed=sd)
# gen_obj_test.fit(x_train, seed=sd)
#
#
# # set optimizer and compile model
# learning_rate = 1e-6
# print("compiling model...")
# sgd = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
# model_VGG16.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
#
# # train the model
# print("training model...")
# epochs=1
# early_stopping = EarlyStopping(monitor='acc', mode='max', patience=10)
#
# # fits the model on batches with real-time data augmentation:
# hist = model_VGG16.fit_generator(gen_obj_training.flow(x_train, y_train, batch_size=32),
#                     steps_per_epoch=len(x_train) / 32, epochs=epochs, verbose=1, callbacks=[early_stopping])
#
# # create save directories if it doesn't exist
# if not os.path.exists(config['model_savepath']):
#     os.makedirs(config['model_savepath'])
# if not os.path.exists(config['plot_path']):
#     os.makedirs(config['plot_path'])
#
# # create a name for the model
# if args['mode'] == 'train_val':
#     modelname = 'shift_{}_{}'.format(args['mode'], args['dataset'])
# if args['mode'] == 'datasets':
#     modelname = 'shift_{}_{}'.format(args['dataset'], args['dataset2'])
#
# # save history
# pd.DataFrame(hist.history).to_csv(os.path.join(config['model_savepath'], '{}_history.csv'.format(modelname)))
#
# # save trained model
# print("saving model...")
# savepath = os.path.join(config['model_savepath'], "{}_VGG16.h5".format(modelname))
# model_VGG16.save(savepath)
#
# # create plot directory if it doesn't exist and plot training progress
# print("saving plots...")
# plotpath = os.path.join(config['plot_path'], "{}_training.png".format(modelname))
# # plot_training(hist, epochs, plotpath)
#
# # now make predictions
# print("evaluating model...")
# preds = model_VGG16.predict_generator(gen_obj_test.flow(x_val, batch_size=1), verbose=1)
#
# # get true labels
# true_labels = np.vstack(y_val)
#
# # plot ROC and calculate AUC
# ROC_AUC(preds, true_labels, config, modelname)
#
# print(np.unique(preds))
#
# unique, counts = np.unique(true_labels, return_counts=True)
# print(dict(zip(unique, counts)))
#
# predictions = np.where(preds >= 0.5, 1, 0).astype(int)
# accuracy = accuracy_score(true_labels, predictions)
#
# unique, counts = np.unique(predictions, return_counts=True)
# print(dict(zip(unique, counts)))
#
# print(accuracy)
# p_val_list = []
# for i in range(len(true_labels)):
#     p_val = binom_test((i+1), len(true_labels), p=0.5)
#     p_val_list.append(p_val)
#
# # plot and save ROC curve
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(range(len(true_labels)), p_val_list, 'r-')
# plt.savefig(os.path.join(config['plot_path'], "pval.png"))
#
# # perform a binomial test and get the p-value
# p_val_test = binomial_test(true_labels, predictions)
# print("p-value: {}".format(p_val_test))
