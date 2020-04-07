import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
from openpyxl import load_workbook, Workbook

# construct argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d',
    '--dataset',
    choices=['ISIC', 'CNMC'],
    required=True,
    help='dataset to use')
parser.add_argument('-m',
    '--mode',
    choices=['acc', 'skAUC'],
    required=True,
    help='mode for plotting, either acc (accuracy) of AUC (Area Under the Curve)')
parser.add_argument('-mod',
    '--modification',
    choices=['add_noise_gaussian', 'add_noise_poisson', 'add_noise_salt_and_pepper',
    'add_noise_speckle', 'grayscale', 'hsv', 'image_rot', 'image_translation',
    'image_zoom', 'imbalance_classes', 'small_clusters', 'small_easy',
    'small_hard', 'small_random'],
    required=True,
    help='for which modification to plot results')
parser.add_argument('-b',
    '--bands',
    choices=['minmax', 'stds', 'ses'],
    default='minmax',
    help='error band type to use for plots, either min/max (minmax), standard deviations (stds) or standard errors (ses)')
args = vars(parser.parse_args())

# initialize important lists and dictionaries
x_pct = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
splits = [2, 3, 4, 5, 6]

# get argument parser variables
mode = args['mode']
dataset = args['dataset']
modification = args['modification']
bands = args['bands']

# set the correct baseline
if mode == 'acc' and dataset == 'ISIC':
    baseline = [0.758] * 10
if mode == 'skAUC' and dataset == 'ISIC':
    baseline = [0.856] * 10

# create a folder to store results if it doesn't exist yet
plotpath = os.path.join(os.path.dirname(os.getcwd()), 'outputs/results/plots')
if not os.path.exists(plotpath):
    os.makedirs(plotpath)

# create empty dictionaries to store results
results_dict = {}

# do for every type of experiment
for type in ['SVM', 'fc', 'fine_tuning']:
    # create empty entry in nested dictionary
    results_dict[type] = {}

    # now do for every split
    for split in splits:
        # read results for the corresponding split
        resultsfile = os.path.join(os.path.dirname(os.getcwd()), 'outputs/results/results_ISIC_{}.xlsx'.format(split))

        # load the results
        results = pd.read_excel(resultsfile, sheet_name=type)

        # get the results for the modification
        result_values = [results.query('source_dataset==@source_dataset')[mode].item() for source_dataset in results['source_dataset'] if modification in source_dataset]

        # store values in a dictionary
        results_dict[type][split] = result_values

    # now we need to calculate the mean values over every split
    # create an anonymous function that gets every nth value for each split and makes a list out of it
    values = lambda n: [results_dict[type][split][n] for split in splits]

    # add the mean values for every value to dictionary
    results_dict[type]['means'] = np.asarray([np.round(np.mean(values(n)),3) for n in range(len(x_pct))])

    # do the same for standard deviations, standard errors and min and max values (for plotting)
    results_dict[type]['stds'] = np.asarray([np.round(np.std(values(n)),3) for n in range(len(x_pct))])
    results_dict[type]['ses'] = np.asarray([np.round(np.std(values(n))/np.sqrt(len(values(n))),3) for n in range(len(x_pct))])
    results_dict[type]['mins'] = np.asarray([np.round(np.min(values(n)),3) for n in range(len(x_pct))])
    results_dict[type]['maxs'] = np.asarray([np.round(np.max(values(n)),3) for n in range(len(x_pct))])

# create cleaner names for plotting
if modification == "add_noise_gaussian": plotname = "Gaussian noise"
if modification == "add_noise_poisson": plotname = "Poisson noise"
if modification == "add_noise_salt_and_pepper": plotname = "Salt & pepper noise"
if modification == "add_noise_speckle": plotname = "Speckle noise"
if modification == "image_rot": plotname = "Image rotation"
if modification == "image_translation": plotname = "Image translation"
if modification == "image_zoom": plotname = "Image zoom"
if modification == "imbalance_classes": plotname = "Class imbalance"
if modification == "grayscale": plotname = "Grayscale"
if modification == "hsv": plotname = "Hue, Saturation, Value"
if modification == "small_random": plotname = "Dataset size, random images"
if modification == "small_easy": plotname = "Dataset size, easy to classify images"
if modification == "small_hard": plotname = "Dataset size, hard to classify images"
if modification == "small_clusters": plotname = "Dataset size, image clusters"

# create figure for plotting
plt.figure()
sns.set(style="darkgrid")

palette = sns.husl_palette(3)
# sns.lineplot(x_pct, means, linewidth=2.5, color=palette[i])

plt.plot(x_pct, baseline, '--', marker='', color='black', linewidth=1.5, alpha=0.6, label="baseline")

for i, type in enumerate(['SVM', 'fc', 'fine_tuning']):
    # set cleaner label names
    if type == 'SVM': label='SVM'
    if type == 'fc': label='FC'
    if type == 'fine_tuning': label='FT'

    # plot the mean values
    plt.errorbar(x_pct, results_dict[type]['means'], fmt='-o', color=palette[i], label=label)

    # plot the right error band
    if bands == "minmax":
        plt.fill_between(x_pct, results_dict[type]['mins'], results_dict[type]['maxs'], color=palette[i], alpha=0.2)
    if bands == "stds":
        plt.fill_between(x_pct, results_dict[type]['means']-results_dict[type]['stds'], results_dict[type]['means']+results_dict[type]['stds'], color=palette[i], alpha=0.2)
    if bands == "ses":
        plt.fill_between(x_pct, results_dict[type]['means']-results_dict[type]['ses'], results_dict[type]['means']+results_dict[type]['ses'], color=palette[i], alpha=0.2)

# set axis labels
plt.xlabel("Fraction of images modified")
if mode == 'skAUC': ylabel = 'AUC'
if mode == 'acc': ylabel = 'Accuracy'
plt.ylabel(ylabel)

# set axes limits
plt.xlim(0.05, 1.05)
plt.ylim(0.45, 1.00)

# set plot title and legend
plt.title(plotname)
plt.legend(loc="lower right")

# save plot
plt.savefig(os.path.join(plotpath, '{}_{}_{}_{}.png'.format(dataset, mode, modification, bands)))
