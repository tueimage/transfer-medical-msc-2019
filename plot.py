import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
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
    required='individual' in sys.argv,
    help='for which modification to plot results')
parser.add_argument('-e',
    '--errortype',
    choices=['minmax', 'stds', 'ses'],
    default='minmax',
    help='error type to use for plots, either min/max (minmax), standard deviations (stds) or standard errors (ses)')
parser.add_argument('-t',
    '--type',
    choices=['individual', 'grouped', 'grouped_3'],
    default='individual',
    help='type of plot to make')
parser.add_argument('-f',
    '--fraction',
    choices=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    type=float,
    default=0.1,
    required='grouped' in sys.argv,
    help='fraction to plot, only when using "grouped" as plot type')
args = vars(parser.parse_args())

# initialize important lists and dictionaries
x_pct = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
splits = [2, 3, 4, 5, 6]

# get argument parser variables
mode = args['mode']
dataset = args['dataset']
modification = args['modification']
errortype = args['errortype']
plottype = args['type']
frac_to_plot = str(args['fraction'])

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

# for individual plots
if plottype == 'individual':
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
        if errortype == "minmax":
            plt.fill_between(x_pct, results_dict[type]['mins'], results_dict[type]['maxs'], color=palette[i], alpha=0.2)
        if errortype == "stds":
            plt.fill_between(x_pct, results_dict[type]['means']-results_dict[type]['stds'], results_dict[type]['means']+results_dict[type]['stds'], color=palette[i], alpha=0.2)
        if errortype == "ses":
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

if plottype == 'grouped':
    # get list of all modifications
    modifications = ['small_easy', 'small_random', 'small_clusters',
    'small_hard', 'imbalance_classes', 'add_noise_speckle', 'add_noise_gaussian',
    'add_noise_poisson', 'add_noise_salt_and_pepper', 'grayscale',
    'hsv', 'image_rot', 'image_zoom', 'image_translation']

    # create empty pandas dataframe
    data = pd.DataFrame(columns=['modification', 'type', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])

    # initialize empty dictionaries to store minimum and maximum values in
    min_values = {'0.1': [], '0.2': [], '0.3': [], '0.4': [], '0.5': [], '0.6': [], '0.7': [], '0.8': [], '0.9': [], '1.0': []}
    max_values = {'0.1': [], '0.2': [], '0.3': [], '0.4': [], '0.5': [], '0.6': [], '0.7': [], '0.8': [], '0.9': [], '1.0': []}
    std_values = {'0.1': [], '0.2': [], '0.3': [], '0.4': [], '0.5': [], '0.6': [], '0.7': [], '0.8': [], '0.9': [], '1.0': []}
    se_values = {'0.1': [], '0.2': [], '0.3': [], '0.4': [], '0.5': [], '0.6': [], '0.7': [], '0.8': [], '0.9': [], '1.0': []}

    # do for every modification
    for mod in modifications:
        # do for every type of experiment
        for type in ['SVM', 'fc', 'fine_tuning']:
            # create empty entry in nested dictionary
            results_dict[type] = {}

            # initialize empty dictionary to save everything in for dataframe
            dict_df = {}

            # create entries in dictionary
            dict_df['modification'] = mod
            if type == 'SVM': dict_df['type'] = 'SVM'
            if type == 'fc': dict_df['type'] = 'FC'
            if type == 'fine_tuning': dict_df['type'] = 'FT'

            # now do for every split
            for split in splits:
                # read results for the corresponding split
                resultsfile = os.path.join(os.path.dirname(os.getcwd()), 'outputs/results/results_ISIC_{}.xlsx'.format(split))

                # load the results
                results = pd.read_excel(resultsfile, sheet_name=type)

                # get the results for the modification
                result_values = [results.query('source_dataset==@source_dataset')[mode].item() for source_dataset in results['source_dataset'] if mod in source_dataset]

                # store values in a dictionary
                results_dict[type][split] = result_values

            # create an anonymous function that gets every nth value for each split and makes a list out of it
            values = lambda n: [results_dict[type][split][n] for split in splits]

            for n in range(len(x_pct)):
                # fractions for modifications that change the dataset size don't mean the same as for the other modifications
                # for these, lower fractions correspond to more modification of the dataset
                # correct for this
                if mod in ['imbalance_classes', 'small_clusters', 'small_easy', 'small_hard', 'small_random']:
                    # add the mean values over every split for every fraction to dictionary
                    dict_df[str(x_pct[n])] = np.round(np.mean(values(len(x_pct)-1-n)),3)

                    # add difference between mean and minimum and maximum value to the corresponding dictionaries
                    min_values[str(x_pct[n])].append(np.round(np.mean(values(len(x_pct)-1-n)) - np.min(values(len(x_pct)-1-n)),3))
                    max_values[str(x_pct[n])].append(np.round(np.max(values(len(x_pct)-1-n)) - np.mean(values(len(x_pct)-1-n)),3))

                    # add standard deviation and standard error to the corresponding dictionaries
                    std_values[str(x_pct[n])].append(np.round(np.std(values(len(x_pct)-1-n)),3))
                    se_values[str(x_pct[n])].append(np.round(np.std(values(len(x_pct)-1-n))/np.sqrt(len(values(len(x_pct)-1-n))),3))
                else:
                    # add the mean values over every split for every fraction to dictionary
                    dict_df[str(x_pct[n])] = np.round(np.mean(values(n)),3)

                    # add difference between mean and minimum and maximum value to the corresponding dictionaries
                    min_values[str(x_pct[n])].append(np.round(np.mean(values(n)) - np.min(values(n)),3))
                    max_values[str(x_pct[n])].append(np.round(np.max(values(n)) - np.mean(values(n)),3))

                    # add standard deviation and standard error to the corresponding dictionaries
                    std_values[str(x_pct[n])].append(np.round(np.std(values(n)),3))
                    se_values[str(x_pct[n])].append(np.round(np.std(values(n))/np.sqrt(len(values(n))),3))

            # add dictionary as a row to the dataframe
            data = data.append(dict_df, ignore_index=True)

    # create a new figure
    plt.figure()
    plt.clf()

    # get color palette
    colors = sns.color_palette("Set2")[:3]

    # plot the mean points
    ax = sns.pointplot(y=frac_to_plot, x='modification', data=data, dodge=0.25, join=False, markers=['o', 's', '^'], scale=0.6, palette=colors, alpha=1.0, hue='type')

    # in order to plot the error bars, we need to find the x,y coordinates for each point
    x_coords = []
    y_coords = []
    for point_pair in ax.collections:
        for x, y in point_pair.get_offsets():
            x_coords.append(x)
            y_coords.append(y)

    # coordinates are taken differently than the points are plotted, rearrange them so they fit
    rearranged_x_coords, rearranged_y_coords = [], []
    for i in range(len(modifications)):
        rearranged_x_coords.append(x_coords[i])
        rearranged_x_coords.append(x_coords[i+len(modifications)])
        rearranged_x_coords.append(x_coords[i+2*len(modifications)])

        rearranged_y_coords.append(y_coords[i])
        rearranged_y_coords.append(y_coords[i+len(modifications)])
        rearranged_y_coords.append(y_coords[i+2*len(modifications)])

    # plot the error bars using the coordinates of the previous points
    if errortype == "minmax":
        ax.errorbar(rearranged_x_coords, rearranged_y_coords, yerr=[min_values[frac_to_plot], max_values[frac_to_plot]], ecolor=colors, fmt=' ')
    if errortype == "stds":
        ax.errorbar(rearranged_x_coords, rearranged_y_coords, yerr=std_values[frac_to_plot], ecolor=colors, fmt=' ', zorder=-1)
    if errortype == "ses":
        ax.errorbar(rearranged_x_coords, rearranged_y_coords, yerr=se_values[frac_to_plot], ecolor=colors, fmt=' ', zorder=-1)

    # save plot
    plt.savefig(os.path.join(plotpath, '{}_{}_{}_{}.png'.format(dataset, mode, errortype, frac_to_plot)))

if plottype == 'grouped_3':
    # create subplots
    fig, axs = plt.subplots(3, sharex=True, sharey=True, gridspec_kw={'hspace': 0})

    plots = 3
    for f in range(plots):
        # do for different fractions
        frac_to_plot = ['0.1', '0.5', '1.0'][f]

        print("Creating subplot for fraction {}...".format(frac_to_plot))

        # get list of all modifications
        modifications = ['small_easy', 'small_random', 'small_clusters',
        'small_hard', 'imbalance_classes', 'add_noise_speckle', 'add_noise_gaussian',
        'add_noise_poisson', 'add_noise_salt_and_pepper', 'grayscale',
        'hsv', 'image_rot', 'image_zoom', 'image_translation']

        # create empty pandas dataframe
        data = pd.DataFrame(columns=['modification', 'type', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])

        # initialize empty dictionaries to store minimum and maximum values in
        min_values = {'0.1': [], '0.2': [], '0.3': [], '0.4': [], '0.5': [], '0.6': [], '0.7': [], '0.8': [], '0.9': [], '1.0': []}
        max_values = {'0.1': [], '0.2': [], '0.3': [], '0.4': [], '0.5': [], '0.6': [], '0.7': [], '0.8': [], '0.9': [], '1.0': []}
        std_values = {'0.1': [], '0.2': [], '0.3': [], '0.4': [], '0.5': [], '0.6': [], '0.7': [], '0.8': [], '0.9': [], '1.0': []}
        se_values = {'0.1': [], '0.2': [], '0.3': [], '0.4': [], '0.5': [], '0.6': [], '0.7': [], '0.8': [], '0.9': [], '1.0': []}

        # do for every modification
        for mod in modifications:
            # do for every type of experiment
            for type in ['SVM', 'fc', 'fine_tuning']:
                # create empty entry in nested dictionary
                results_dict[type] = {}

                # initialize empty dictionary to save everything in for dataframe
                dict_df = {}

                # create entries in dictionary
                dict_df['modification'] = mod
                if type == 'SVM': dict_df['type'] = 'SVM'
                if type == 'fc': dict_df['type'] = 'FC'
                if type == 'fine_tuning': dict_df['type'] = 'FT'

                # now do for every split
                for split in splits:
                    # read results for the corresponding split
                    resultsfile = os.path.join(os.path.dirname(os.getcwd()), 'outputs/results/results_ISIC_{}.xlsx'.format(split))

                    # load the results
                    results = pd.read_excel(resultsfile, sheet_name=type)

                    # get the results for the modification
                    result_values = [results.query('source_dataset==@source_dataset')[mode].item() for source_dataset in results['source_dataset'] if mod in source_dataset]

                    # store values in a dictionary
                    results_dict[type][split] = result_values

                # create an anonymous function that gets every nth value for each split and makes a list out of it
                values = lambda n: [results_dict[type][split][n] for split in splits]

                for n in range(len(x_pct)):
                    # fractions for modifications that change the dataset size don't mean the same as for the other modifications
                    # for these, lower fractions correspond to more modification of the dataset
                    # correct for this
                    if mod in ['imbalance_classes', 'small_clusters', 'small_easy', 'small_hard', 'small_random']:
                        # add the mean values over every split for every fraction to dictionary
                        dict_df[str(x_pct[n])] = np.round(np.mean(values(len(x_pct)-1-n)),3)

                        # add difference between mean and minimum and maximum value to the corresponding dictionaries
                        min_values[str(x_pct[n])].append(np.round(np.mean(values(len(x_pct)-1-n)) - np.min(values(len(x_pct)-1-n)),3))
                        max_values[str(x_pct[n])].append(np.round(np.max(values(len(x_pct)-1-n)) - np.mean(values(len(x_pct)-1-n)),3))

                        # add standard deviation and standard error to the corresponding dictionaries
                        std_values[str(x_pct[n])].append(np.round(np.std(values(len(x_pct)-1-n)),3))
                        se_values[str(x_pct[n])].append(np.round(np.std(values(len(x_pct)-1-n))/np.sqrt(len(values(len(x_pct)-1-n))),3))
                    else:
                        # add the mean values over every split for every fraction to dictionary
                        dict_df[str(x_pct[n])] = np.round(np.mean(values(n)),3)

                        # add difference between mean and minimum and maximum value to the corresponding dictionaries
                        min_values[str(x_pct[n])].append(np.round(np.mean(values(n)) - np.min(values(n)),3))
                        max_values[str(x_pct[n])].append(np.round(np.max(values(n)) - np.mean(values(n)),3))

                        # add standard deviation and standard error to the corresponding dictionaries
                        std_values[str(x_pct[n])].append(np.round(np.std(values(n)),3))
                        se_values[str(x_pct[n])].append(np.round(np.std(values(n))/np.sqrt(len(values(n))),3))

                # add dictionary as a row to the dataframe
                data = data.append(dict_df, ignore_index=True)

        # get color palette
        colors = sns.color_palette("Set2")[:3]

        # plot the mean points
        ax = sns.pointplot(y=frac_to_plot, x='modification', data=data, dodge=0.25, join=False, markers=['o', 's', '^'], scale=0.6, palette=colors, alpha=1.0, hue='type', ax=axs[f])

        # in order to plot the error bars, we need to find the x,y coordinates for each point
        x_coords = []
        y_coords = []
        for point_pair in axs[f].collections:
            for x, y in point_pair.get_offsets():
                x_coords.append(x)
                y_coords.append(y)

        # coordinates are taken differently than the points are plotted, rearrange them so they fit
        rearranged_x_coords, rearranged_y_coords = [], []
        for i in range(len(modifications)):
            rearranged_x_coords.append(x_coords[i])
            rearranged_x_coords.append(x_coords[i+len(modifications)])
            rearranged_x_coords.append(x_coords[i+2*len(modifications)])

            rearranged_y_coords.append(y_coords[i])
            rearranged_y_coords.append(y_coords[i+len(modifications)])
            rearranged_y_coords.append(y_coords[i+2*len(modifications)])

        # plot the error bars using the coordinates of the previous points
        if errortype == "minmax":
            ax.errorbar(rearranged_x_coords, rearranged_y_coords, yerr=[min_values[frac_to_plot], max_values[frac_to_plot]], ecolor=colors, fmt=' ')
        if errortype == "stds":
            ax.errorbar(rearranged_x_coords, rearranged_y_coords, yerr=std_values[frac_to_plot], ecolor=colors, fmt=' ', zorder=-1)
        if errortype == "ses":
            ax.errorbar(rearranged_x_coords, rearranged_y_coords, yerr=se_values[frac_to_plot], ecolor=colors, fmt=' ', zorder=-1)

        # only place legend on last subplot and in the right place
        if f != plots-1:
            axs[f].get_legend().remove()
        else:
            axs[f].legend(loc="lower right")

        # set y labels
        ylabels = ["Lowest $\delta$", "Medium $\delta$", "Highest $\delta$"]
        for f in range(plots):
            axs[f].set_ylabel(ylabels[f])

    # hide x labels and tick labels for all but bottom plot and set grid lines
    for ax in axs:
        ax.label_outer()
        ax.grid(True, which='major', axis='y')

    # set cleaner labels for x axis
    axs[-1].set_xticklabels(['A \n Size\neasy', 'B \n Size\nrandom', 'C \n Size\nclusters', 'D \n Size\nhard', 'E \n Class imbalance', 'F \n Speckle\nnoise', 'G \n Gaussian\nnoise', 'H \n Poisson\nnoise', 'I \n Salt & pepper\nnoise', 'J \n Grayscale', 'K \n Hue,\nSaturation,\nValue', 'L \n Rotation', 'M \n Zoom', 'N \n Translation'], fontweight='bold')

    # save plot
    plt.savefig(os.path.join(plotpath, '{}_{}_{}_grouped-3.png'.format(dataset, mode, errortype)))
