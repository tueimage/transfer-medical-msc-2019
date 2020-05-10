import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
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
    required='individual' in sys.argv or 'grouped' in sys.argv or 'grouped_3' in sys.argv or 'grouped_type' in sys.argv,
    help='mode for plotting, either acc (accuracy) of AUC (Area Under the Curve)')
parser.add_argument('-mod',
    '--modification',
    choices=['add_noise_gaussian', 'add_noise_poisson', 'add_noise_salt_and_pepper',
    'add_noise_speckle', 'grayscale', 'hsv', 'image_rot', 'image_translation',
    'image_zoom', 'imbalance_classes', 'small_clusters', 'small_easy',
    'small_hard', 'small_random'],
    required='individual' in sys.argv or 'shift_AUC' in sys.argv,
    help='for which modification to plot results')
parser.add_argument('-e',
    '--errortype',
    choices=['minmax', 'stds', 'ses'],
    default='minmax',
    help='error type to use for plots, either min/max (minmax), standard deviations (stds) or standard errors (ses)')
parser.add_argument('-t',
    '--type',
    choices=['individual', 'grouped', 'grouped_3', 'shift', 'shift_size', 'grouped_type', 'shift_AUC', 'heatmap'],
    default='individual',
    help='type of plot to make')
parser.add_argument('-tt',
    '--transfertype',
    choices=['SVM', 'fc', 'fine_tuning'],
    required='heatmap' in sys.argv,
    help='transfer method to make heatmap for')
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

if mode == 'acc' and dataset == 'CNMC':
    baseline = [0.889] * 10
if mode == 'skAUC' and dataset == 'CNMC':
    baseline = [0.951] * 10

# create a folder to store results if it doesn't exist yet
plotpath = os.path.join(os.path.dirname(os.getcwd()), 'outputs/results/plots')
if not os.path.exists(plotpath):
    os.makedirs(plotpath)

# create empty dictionaries to store results
results_dict = {}
if plottype == 'heatmap':
    # get list of all modifications
    modifications = ['small_easy', 'small_random', 'small_clusters',
    'small_hard', 'imbalance_classes', 'add_noise_gaussian',
    'grayscale', 'hsv', 'image_rot', 'image_zoom', 'image_translation']

    # create empty pandas dataframe
    data = pd.DataFrame(columns=['modification', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])

    # initialize empty dictionaries to store minimum and maximum values in
    min_values = {'0.1': [], '0.2': [], '0.3': [], '0.4': [], '0.5': [], '0.6': [], '0.7': [], '0.8': [], '0.9': [], '1.0': []}
    max_values = {'0.1': [], '0.2': [], '0.3': [], '0.4': [], '0.5': [], '0.6': [], '0.7': [], '0.8': [], '0.9': [], '1.0': []}
    std_values = {'0.1': [], '0.2': [], '0.3': [], '0.4': [], '0.5': [], '0.6': [], '0.7': [], '0.8': [], '0.9': [], '1.0': []}
    se_values = {'0.1': [], '0.2': [], '0.3': [], '0.4': [], '0.5': [], '0.6': [], '0.7': [], '0.8': [], '0.9': [], '1.0': []}

    # set type
    type = args['transfertype']

    # do for every modification
    for mod in modifications:

        # create empty entry in nested dictionary
        results_dict = {}

        # initialize empty dictionary to save everything in for dataframe
        dict_df = {}

        # create entries in dictionary
        dict_df['modification'] = mod

        # now do for every split
        for split in splits:
            # read results for the corresponding split
            resultsfile = os.path.join(os.path.dirname(os.getcwd()), 'outputs/results/results_{}_{}_all.xlsx'.format(dataset, split))

            # load the results
            results = pd.read_excel(resultsfile, sheet_name=type)

            # get the results for the modification
            result_values = [results.query('source_dataset==@source_dataset')[mode].item() for source_dataset in results['source_dataset'] if mod in source_dataset]

            # store values in a dictionary
            results_dict[split] = result_values

        # create an anonymous function that gets every nth value for each split and makes a list out of it
        values = lambda n: [results_dict[split][n] for split in splits]

        # create a function that filters the values from the above values function
        # split values that are all 0.5 stay all 0.5, otherwise only keep the values that are not 0.5
        filter_values = lambda vals: list(filter(lambda a: a != 0.5, vals)) if (sum(vals) != 2.5) else vals

        for n in range(len(x_pct)):
            # fractions for modifications that change the dataset size don't mean the same as for the other modifications
            # for these, lower fractions correspond to more modification of the dataset
            # correct for this
            if mod in ['imbalance_classes', 'small_clusters', 'small_easy', 'small_hard', 'small_random']:
                # add the mean values over every split for every fraction to dictionary
                dict_df[str(x_pct[n])] = np.round(np.mean(filter_values(values(len(x_pct)-1-n))),3)

                # add difference between mean and minimum and maximum value to the corresponding dictionaries
                min_values[str(x_pct[n])].append(np.round(np.mean(filter_values(values(len(x_pct)-1-n))) - np.min(filter_values(values(len(x_pct)-1-n))),3))
                max_values[str(x_pct[n])].append(np.round(np.max(filter_values(values(len(x_pct)-1-n))) - np.mean(filter_values(values(len(x_pct)-1-n))),3))

                # add standard deviation and standard error to the corresponding dictionaries
                std_values[str(x_pct[n])].append(np.round(np.std(filter_values(values(len(x_pct)-1-n))),3))
                se_values[str(x_pct[n])].append(np.round(np.std(filter_values(values(len(x_pct)-1-n)))/np.sqrt(len(filter_values(values(len(x_pct)-1-n)))),3))
            else:
                # add the mean values over every split for every fraction to dictionary
                dict_df[str(x_pct[n])] = np.round(np.mean(filter_values(values(n))),3)

                # add difference between mean and minimum and maximum value to the corresponding dictionaries
                min_values[str(x_pct[n])].append(np.round(np.mean(filter_values(values(n))) - np.min(filter_values(values(n))),3))
                max_values[str(x_pct[n])].append(np.round(np.max(filter_values(values(n))) - np.mean(filter_values(values(n))),3))

                # add standard deviation and standard error to the corresponding dictionaries
                std_values[str(x_pct[n])].append(np.round(np.std(filter_values(values(n))),3))
                se_values[str(x_pct[n])].append(np.round(np.std(filter_values(values(n)))/np.sqrt(len(filter_values(values(n)))),3))

        # add dictionary as a row to the dataframe
        data = data.append(dict_df, ignore_index=True)

    data = data.set_index('modification')

    # create cleaner plot labels
    data.rename(index={'add_noise_gaussian': 'Gaussian noise', 'image_rot': 'Image rotation',
        'image_translation': 'Image translation', 'image_zoom': 'Image zoom',
        'imbalance_classes': 'Class imbalance', 'grayscale': 'Grayscale',
        'hsv': 'Hue, Saturation, Value', 'small_random': 'Size, random',
        'small_easy': 'Size, easy', 'small_hard': 'Size, hard',
        'small_clusters': 'Size, clusters'}, inplace=True)

    fig, (ax1, ax2) = plt.subplots(2,1)

    # create some space between subplots for x-axis title
    fig.tight_layout(pad=1.5)

    # create heatmaps
    im = sns.heatmap(data[:5], annot=True, cmap='RdYlGn', vmin=0.5, vmax=1.0, cbar=False, ax=ax1)
    sns.heatmap(data[5:], annot=True, cmap='RdYlGn', vmin=0.5, vmax=1.0, cbar=False, ax=ax2)

    # created shared colorbar
    mappable = im.get_children()[0]
    plt.colorbar(mappable, ax=[ax1,ax2], orientation='vertical', drawedges=False, label='AUC')

    # set cleaner name for title
    if type == 'SVM':
        plottitle = 'SVM'
    if type == 'fc':
        plottitle = 'Fully Connected layers'
    if type == 'fine_tuning':
        plottitle = 'Finetuning'

    if dataset == 'ISIC':
        datasetname = 'ISIC'
    if dataset == 'CNMC':
        datasetname = 'C-NMC'

    # set plot title
    plt.suptitle('{}, {}'.format(datasetname, plottitle))

    # set plot axes
    ax1.set_ylabel('')
    ax1.set_xlabel('Fraction of dataset images used')
    ax1.set_xticklabels(['1.0', '0.9', '0.8', '0.7', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1'])
    ax2.set_ylabel('')
    ax2.set_xlabel('Fraction of dataset images modified')

    # save figure
    plt.savefig(os.path.join(plotpath, 'heatmap_NOOUTL_{}_{}.png'.format(dataset, type)), bbox_inches='tight')

if plottype == 'shift_AUC':
    # do for every type of experiment
    for type in ['SVM', 'fc', 'fine_tuning']:
        # create empty entry in nested dictionary
        results_dict[type] = {}

        # now do for every split
        for split in splits:
            # read results for the corresponding split
            resultsfile = os.path.join(os.path.dirname(os.getcwd()), 'outputs/results/results_{}_{}_all.xlsx'.format(dataset, split))

            # load the results
            results = pd.read_excel(resultsfile, sheet_name=type)

            # get the results for the modification
            result_values = [results.query('source_dataset==@source_dataset')[mode].item() for source_dataset in results['source_dataset'] if modification in source_dataset]

            # store values in a dictionary
            results_dict[type][split] = result_values

        # now we need to calculate the mean values over every split
        # create an anonymous function that gets every nth value for each split and makes a list out of it
        values = lambda n: [results_dict[type][split][n] for split in splits]

        # create a function that filters the values from the above values function
        # split values that are all 0.5 stay all 0.5, otherwise only keep the values that are not 0.5
        filter_values = lambda vals: list(filter(lambda a: a != 0.5, vals)) if (sum(vals) != 2.5) else vals

        # add the mean values for every value to dictionary
        results_dict[type]['means'] = np.asarray([np.round(np.mean(values(n)),3) for n in range(len(x_pct))])

        # do the same for standard deviations, standard errors and min and max values (for plotting)
        results_dict[type]['stds'] = np.asarray([np.round(np.std(values(n)),3) for n in range(len(x_pct))])
        results_dict[type]['ses'] = np.asarray([np.round(np.std(values(n))/np.sqrt(len(values(n))),3) for n in range(len(x_pct))])
        results_dict[type]['mins'] = np.asarray([np.round(np.min(values(n)),3) for n in range(len(x_pct))])
        results_dict[type]['maxs'] = np.asarray([np.round(np.max(values(n)),3) for n in range(len(x_pct))])

    # add the shift results to the dictionary
    results_dict['shift'] = {}

    for split in splits:
        # read results for the corresponding split
        resultsfile_shift = os.path.join(os.path.dirname(os.getcwd()), 'outputs/results/results_shift_{}_{}_all.xlsx'.format(args['dataset'], split))

        # load the results
        results_shift = pd.read_excel(resultsfile_shift, sheet_name='detect_shift')

        # get the results for the modification
        result_values_shift = [results_shift.query('dataset2==@dataset2')['absAUC'].item() for dataset2 in results_shift['dataset2'] if modification in dataset2]

        # store values in the dictionary
        results_dict['shift'][split] = result_values_shift

    # create an anonymous function that gets every nth value for each split and makes a list out of it
    values = lambda n: [results_dict['shift'][split][n] for split in splits]

    # add the mean values for every value to dictionary
    results_dict['shift']['means'] = np.asarray([np.round(np.mean(values(n)),3) for n in range(len(x_pct))])

    # do the same for standard deviations, standard errors and min and max values (for plotting)
    results_dict['shift']['stds'] = np.asarray([np.round(np.std(values(n)),3) for n in range(len(x_pct))])
    results_dict['shift']['ses'] = np.asarray([np.round(np.std(values(n))/np.sqrt(len(values(n))),3) for n in range(len(x_pct))])
    results_dict['shift']['mins'] = np.asarray([np.round(np.min(values(n)),3) for n in range(len(x_pct))])
    results_dict['shift']['maxs'] = np.asarray([np.round(np.max(values(n)),3) for n in range(len(x_pct))])

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

    # get nice colors for plotting
    colors = ['#85cbcf', '#3984b6', '#1d2e81']

    # create anonymous function to convert hex values to rgb format
    hex_to_rgb = lambda hex: tuple(int(hex.lstrip('#')[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    # convert the color palette to rgb
    rgb_colors = [hex_to_rgb(colors[i]) for i in range(len(colors))]

    fig, ax = plt.subplots()

    for i, type in enumerate(['SVM', 'fc', 'fine_tuning']):
        # set cleaner label names
        if type == 'SVM': label='SVM'
        if type == 'fc': label='FC'
        if type == 'fine_tuning': label='FT'

        # plot the mean values
        ax.errorbar(results_dict['shift']['means'], results_dict[type]['means'], fmt='o', color=rgb_colors[i], label=label)

        z = np.polyfit(results_dict['shift']['means'], results_dict[type]['means'], 1)
        p = np.poly1d(z)
        ax.plot(results_dict['shift']['means'],p(results_dict['shift']['means']),color=rgb_colors[i], ls='--')

        # plt.xscale('log')
        # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        # ax.get_xaxis().get_major_formatter().labelOnlyBase = False
        # ax.get_xaxis().set_tick_params(which='minor', size=0)
        # ax.get_xaxis().set_tick_params(which='minor', width=0)
        # sns.lmplot(x=results_dict['shift']['means'], y=results_dict[type]['means'], ci=None)

        # plot the right error band
        # if errortype == "minmax":
        #     plt.fill_between(results_dict['shift']['means'], results_dict[type]['mins'], results_dict[type]['maxs'], color=rgb_colors[i], alpha=0.2)
        # if errortype == "stds":
        #     plt.fill_between(results_dict['shift']['means'], results_dict[type]['means']-results_dict[type]['stds'], results_dict[type]['means']+results_dict[type]['stds'], color=rgb_colors[i], alpha=0.2)
        # if errortype == "ses":
        #     plt.fill_between(results_dict['shift']['means'], results_dict[type]['means']-results_dict[type]['ses'], results_dict[type]['means']+results_dict[type]['ses'], color=rgb_colors[i], alpha=0.2)

    # set axis labels
    plt.xlabel("Dataset shift ($\Delta$AUC)")
    if mode == 'skAUC': ylabel = 'AUC'
    if mode == 'acc': ylabel = 'Accuracy'
    plt.ylabel(ylabel)

    # set axes limits
    # plt.xlim(0.05, 1.05)
    plt.ylim(0.45, 1.00)

    # if 'small' in modification or 'imbalance' in modification:
    #     plt.xlim(1.05, 0.05)
    #     plt.xlabel("Fraction of dataset used")

    # set plot title and legend
    plt.title(plotname)
    plt.legend(loc="upper right")

    # plt.show()

    plt.savefig(os.path.join(plotpath, 'AUC_vs_shift_{}_{}_{}.png'.format(dataset, modification, args['errortype'])))

if plottype == 'shift':
    # for which modifications to plot
    # modifications = ['small_easy', 'small_random', 'small_clusters',
    # 'small_hard', 'imbalance_classes', 'add_noise_speckle', 'add_noise_gaussian', 'grayscale',
    # 'hsv', 'image_rot', 'image_zoom', 'image_translation']

    # modifications = ['small_easy', 'small_random', 'small_clusters',
    # 'small_hard', 'imbalance_classes']

    # modifications = ['small_random', 'small_hard', 'small_easy', 'imbalance_classes', 'small_clusters']

    modifications = ['hsv', 'image_rot', 'image_zoom', 'add_noise_gaussian',
    'grayscale', 'image_translation']

    # create empty pandas dataframe
    data = pd.DataFrame(columns=['modification', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])

    # initialize empty dictionaries to store minimum and maximum values in
    min_values = {'0.1': [], '0.2': [], '0.3': [], '0.4': [], '0.5': [], '0.6': [], '0.7': [], '0.8': [], '0.9': [], '1.0': []}
    max_values = {'0.1': [], '0.2': [], '0.3': [], '0.4': [], '0.5': [], '0.6': [], '0.7': [], '0.8': [], '0.9': [], '1.0': []}
    std_values = {'0.1': [], '0.2': [], '0.3': [], '0.4': [], '0.5': [], '0.6': [], '0.7': [], '0.8': [], '0.9': [], '1.0': []}
    se_values = {'0.1': [], '0.2': [], '0.3': [], '0.4': [], '0.5': [], '0.6': [], '0.7': [], '0.8': [], '0.9': [], '1.0': []}

    # do for every modification
    for mod in modifications:
        # create empty entry in nested dictionary
        results_dict[mod] = {}

        # initialize empty dictionary to save everything in for dataframe
        dict_df = {}

        # create entries in dictionary
        dict_df['modification'] = mod
        # if type == 'absAUC': dict_df['type'] = 'dAUC'

        # now do for every split
        for split in splits:
            # read results for the corresponding split
            resultsfile = os.path.join(os.path.dirname(os.getcwd()), 'outputs/results/results_shift_{}_{}_all.xlsx'.format(args['dataset'], split))

            # load the results
            results = pd.read_excel(resultsfile, sheet_name='detect_shift')

            # get the results for the modification
            result_values = [results.query('dataset2==@dataset2')['absAUC'].item() for dataset2 in results['dataset2'] if mod in dataset2]

            # store values in a dictionary
            results_dict[mod][split] = result_values

        # create an anonymous function that gets every nth value for each split and makes a list out of it
        values = lambda n: [results_dict[mod][split][n] for split in splits]

        # add the mean values for every value to dictionary
        results_dict[mod]['means'] = np.asarray([np.round(np.mean(values(n)),3) for n in range(len(x_pct))])

        # do the same for standard deviations, standard errors and min and max values (for plotting)
        results_dict[mod]['stds'] = np.asarray([np.round(np.std(values(n)),3) for n in range(len(x_pct))])
        results_dict[mod]['ses'] = np.asarray([np.round(np.std(values(n))/np.sqrt(len(values(n))),3) for n in range(len(x_pct))])
        results_dict[mod]['mins'] = np.asarray([np.round(np.min(values(n)),3) for n in range(len(x_pct))])
        results_dict[mod]['maxs'] = np.asarray([np.round(np.max(values(n)),3) for n in range(len(x_pct))])

    # create figure for plotting
    plt.figure()
    sns.set(style="darkgrid")

    palette = sns.husl_palette(len(modifications))

    # color palette with 5,6 or 12 colors
    if len(modifications) == 5:
        # colors = ['#b7dfcb', '#5abad1', '#3984b6', '#264992', '#161f63']
        colors = ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641']
    if len(modifications) == 6:
        # colors = ['#bee0cc', '#70c3d0', '#419dc5', '#316ba7', '#223b89', '#151e5e']
        colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
    # if len(modifications) == 12:
    #     colors = ['#dcecc9', '#b3ddcc', '#8acdce', '#62bed2', '#46aace', '#3d91be', '#3577ae', '#2d5e9e', '#24448e', '#1c2b7f', '#162065', '#11174b']

    # create anonymous function to convert hex values to rgb format
    hex_to_rgb = lambda hex: tuple(int(hex.lstrip('#')[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    # convert the color palette to rgb
    rgb_colors = [hex_to_rgb(colors[i]) for i in range(len(colors))]

    # sns.lineplot(x_pct, means, linewidth=2.5, color=palette[i])

    # plt.plot(x_pct, baseline, '--', marker='', color='black', linewidth=1.5, alpha=0.6, label="baseline")

    for i, mod in enumerate(modifications):
        # create cleaner names for plotting
        if mod == "add_noise_gaussian": plotname = "Gaussian noise"
        if mod == "add_noise_poisson": plotname = "Poisson noise"
        if mod == "add_noise_salt_and_pepper": plotname = "Salt & pepper noise"
        if mod == "add_noise_speckle": plotname = "Speckle noise"
        if mod == "image_rot": plotname = "Image rotation"
        if mod == "image_translation": plotname = "Image translation"
        if mod == "image_zoom": plotname = "Image zoom"
        if mod == "imbalance_classes": plotname = "Class imbalance"
        if mod == "grayscale": plotname = "Grayscale"
        if mod == "hsv": plotname = "Hue, Saturation, Value"
        if mod == "small_random": plotname = "Dataset size, random images"
        if mod == "small_easy": plotname = "Dataset size, easy to classify images"
        if mod == "small_hard": plotname = "Dataset size, hard to classify images"
        if mod == "small_clusters": plotname = "Dataset size, image clusters"

        # plot the mean values
        plt.errorbar(x_pct, results_dict[mod]['means'], fmt='-o', color=rgb_colors[i], label=plotname)

        # plot the right error band
        if errortype == "minmax":
            plt.fill_between(x_pct, results_dict[mod]['mins'], results_dict[mod]['maxs'], color=rgb_colors[i], alpha=0.2)
        if errortype == "stds":
            plt.fill_between(x_pct, results_dict[mod]['means']-results_dict[mod]['stds'], results_dict[mod]['means']+results_dict[mod]['stds'], color=rgb_colors[i], alpha=0.2)
        if errortype == "ses":
            plt.fill_between(x_pct, results_dict[mod]['means']-results_dict[mod]['ses'], results_dict[mod]['means']+results_dict[mod]['ses'], color=rgb_colors[i], alpha=0.2)

    # set axis labels
    plt.xlabel("Fraction of images modified")
    plt.ylabel('$\Delta$AUC')

    # set axes limits
    # plt.xlim(0, 1.0)
    plt.ylim(0.0, 0.5)

    # plt.xlim(1.05, 0.05)
    # plt.ylim(0.0, 0.25)
    plt.xlabel("Fraction of dataset used")

    # set plot title and legend
    # plt.title("Shift for different modifications")
    plt.legend(loc="upper left")

    # save plot
    # plt.savefig(os.path.join(plotpath, 'shift_size_{}_{}_{}_025.png'.format(dataset, args['errortype'], len(modifications))))
    # plt.savefig(os.path.join(plotpath, 'shift_{}_{}_{}.png'.format(dataset, args['errortype'], len(modifications))))

    plt.show()

if plottype == 'shift_size':
    # for which modifications to plot
    # modifications = ['small_easy', 'small_random', 'small_clusters',
    # 'small_hard', 'imbalance_classes', 'add_noise_speckle', 'add_noise_gaussian', 'grayscale',
    # 'hsv', 'image_rot', 'image_zoom', 'image_translation']

    modifications = ['small_random', 'small_hard', 'small_easy', 'small_clusters',
    'imbalance_classes']

    # modifications = ['hsv', 'image_rot', 'image_zoom', 'add_noise_gaussian',
    # 'grayscale', 'image_translation']

    # create empty pandas dataframe
    data = pd.DataFrame(columns=['modification', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])

    # initialize empty dictionaries to store minimum and maximum values in
    min_values = {'0.1': [], '0.2': [], '0.3': [], '0.4': [], '0.5': [], '0.6': [], '0.7': [], '0.8': [], '0.9': [], '1.0': []}
    max_values = {'0.1': [], '0.2': [], '0.3': [], '0.4': [], '0.5': [], '0.6': [], '0.7': [], '0.8': [], '0.9': [], '1.0': []}
    std_values = {'0.1': [], '0.2': [], '0.3': [], '0.4': [], '0.5': [], '0.6': [], '0.7': [], '0.8': [], '0.9': [], '1.0': []}
    se_values = {'0.1': [], '0.2': [], '0.3': [], '0.4': [], '0.5': [], '0.6': [], '0.7': [], '0.8': [], '0.9': [], '1.0': []}

    # do for every modification
    for mod in modifications:
        # create empty entry in nested dictionary
        results_dict[mod] = {}

        # initialize empty dictionary to save everything in for dataframe
        dict_df = {}

        # create entries in dictionary
        dict_df['modification'] = mod
        # if type == 'absAUC': dict_df['type'] = 'dAUC'

        # now do for every split
        for split in splits:
            # read results for the corresponding split
            resultsfile = os.path.join(os.path.dirname(os.getcwd()), 'outputs/results/results_shift_{}_{}_all.xlsx'.format(args['dataset'], split))

            # load the results
            results = pd.read_excel(resultsfile, sheet_name='detect_shift')

            # get the results for the modification
            result_values = [results.query('dataset2==@dataset2')['absAUC'].item() for dataset2 in results['dataset2'] if mod in dataset2]

            # store values in a dictionary
            results_dict[mod][split] = result_values

        # create an anonymous function that gets every nth value for each split and makes a list out of it
        values = lambda n: [results_dict[mod][split][n] for split in splits]

        # add the mean values for every value to dictionary
        results_dict[mod]['means'] = np.asarray([np.round(np.mean(values(n)),3) for n in range(len(x_pct))])

        # do the same for standard deviations, standard errors and min and max values (for plotting)
        results_dict[mod]['stds'] = np.asarray([np.round(np.std(values(n)),3) for n in range(len(x_pct))])
        results_dict[mod]['ses'] = np.asarray([np.round(np.std(values(n))/np.sqrt(len(values(n))),3) for n in range(len(x_pct))])
        results_dict[mod]['mins'] = np.asarray([np.round(np.min(values(n)),3) for n in range(len(x_pct))])
        results_dict[mod]['maxs'] = np.asarray([np.round(np.max(values(n)),3) for n in range(len(x_pct))])

    # create figure for plotting
    plt.figure()
    sns.set(style="darkgrid")

    palette = sns.husl_palette(len(modifications))

    # color palette with 5,6 or 12 colors
    if len(modifications) == 5:
        colors = ['#b7dfcb', '#5abad1', '#3984b6', '#264992', '#161f63']
    if len(modifications) == 6:
        colors = ['#bee0cc', '#70c3d0', '#419dc5', '#316ba7', '#223b89', '#151e5e']
    if len(modifications) == 12:
        colors = ['#dcecc9', '#b3ddcc', '#8acdce', '#62bed2', '#46aace', '#3d91be', '#3577ae', '#2d5e9e', '#24448e', '#1c2b7f', '#162065', '#11174b']

    # create anonymous function to convert hex values to rgb format
    hex_to_rgb = lambda hex: tuple(int(hex.lstrip('#')[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    # convert the color palette to rgb
    rgb_colors = [hex_to_rgb(colors[i]) for i in range(len(colors))]

    # sns.lineplot(x_pct, means, linewidth=2.5, color=palette[i])

    # plt.plot(x_pct, baseline, '--', marker='', color='black', linewidth=1.5, alpha=0.6, label="baseline")

    for i, mod in enumerate(modifications):
        # create cleaner names for plotting
        if mod == "add_noise_gaussian": plotname = "Gaussian noise"
        if mod == "add_noise_poisson": plotname = "Poisson noise"
        if mod == "add_noise_salt_and_pepper": plotname = "Salt & pepper noise"
        if mod == "add_noise_speckle": plotname = "Speckle noise"
        if mod == "image_rot": plotname = "Image rotation"
        if mod == "image_translation": plotname = "Image translation"
        if mod == "image_zoom": plotname = "Image zoom"
        if mod == "imbalance_classes": plotname = "Class imbalance"
        if mod == "grayscale": plotname = "Grayscale"
        if mod == "hsv": plotname = "Hue, Saturation, Value"
        if mod == "small_random": plotname = "Dataset size, random images"
        if mod == "small_easy": plotname = "Dataset size, easy to classify images"
        if mod == "small_hard": plotname = "Dataset size, hard to classify images"
        if mod == "small_clusters": plotname = "Dataset size, image clusters"

        # plot the mean values
        plt.errorbar(x_pct, results_dict[mod]['means'], fmt='-o', color=rgb_colors[i], label=plotname)

        # plot the right error band
        if errortype == "minmax":
            plt.fill_between(x_pct, results_dict[mod]['mins'], results_dict[mod]['maxs'], color=rgb_colors[i], alpha=0.2)
        if errortype == "stds":
            plt.fill_between(x_pct, (results_dict[mod]['means']-results_dict[mod]['stds']), (results_dict[mod]['means']+results_dict[mod]['stds']), color=rgb_colors[i], alpha=0.2)
        if errortype == "ses":
            plt.fill_between(x_pct, (results_dict[mod]['means']-results_dict[mod]['ses']), (results_dict[mod]['means']+results_dict[mod]['ses']), color=rgb_colors[i], alpha=0.2)

    # set axis labels
    plt.xlabel("Fraction of images modified")
    plt.ylabel('$\Delta$AUC')

    # set axes limits
    plt.xlim(1.05, 0.05)
    plt.ylim(0.0, 0.3)

    # set plot title and legend
    # plt.title("Shift for different modifications")
    plt.legend(loc="upper left")

    # save plot
    plt.savefig(os.path.join(plotpath, 'shift-size_{}_{}_{}-0.3.png'.format(dataset, args['errortype'], len(modifications))))

    plt.show()

# for individual plots
if plottype == 'individual':
    # do for every type of experiment
    for type in ['SVM', 'fc', 'fine_tuning']:
        # create empty entry in nested dictionary
        results_dict[type] = {}

        # now do for every split
        for split in splits:
            # read results for the corresponding split
            resultsfile = os.path.join(os.path.dirname(os.getcwd()), 'outputs/results/results_{}_{}_all.xlsx'.format(dataset, split))

            # load the results
            results = pd.read_excel(resultsfile, sheet_name=type)

            # get the results for the modification
            result_values = [results.query('source_dataset==@source_dataset')[mode].item() for source_dataset in results['source_dataset'] if modification in source_dataset]

            # store values in a dictionary
            results_dict[type][split] = result_values

        # now we need to calculate the mean values over every split
        # create an anonymous function that gets every nth value for each split and makes a list out of it
        values = lambda n: [results_dict[type][split][n] for split in splits]

        # create a function that filters the values from the above values function
        # split values that are all 0.5 stay all 0.5, otherwise only keep the values that are not 0.5
        filter_values = lambda vals: list(filter(lambda a: a != 0.5, vals)) if (sum(vals) != 2.5) else vals

        # add the mean values for every value to dictionary
        results_dict[type]['means'] = np.asarray([np.round(np.mean(filter_values(values(n))),3) for n in range(len(x_pct))])

        # do the same for standard deviations, standard errors and min and max values (for plotting)
        results_dict[type]['stds'] = np.asarray([np.round(np.std(filter_values(values(n))),3) for n in range(len(x_pct))])
        results_dict[type]['ses'] = np.asarray([np.round(np.std(filter_values(values(n)))/np.sqrt(len(filter_values(values(n)))),3) for n in range(len(x_pct))])
        results_dict[type]['mins'] = np.asarray([np.round(np.min(filter_values(values(n))),3) for n in range(len(x_pct))])
        results_dict[type]['maxs'] = np.asarray([np.round(np.max(filter_values(values(n))),3) for n in range(len(x_pct))])

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

    # get nice colors for plotting
    colors = ['#85cbcf', '#3984b6', '#1d2e81']

    # create anonymous function to convert hex values to rgb format
    hex_to_rgb = lambda hex: tuple(int(hex.lstrip('#')[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    # convert the color palette to rgb
    rgb_colors = [hex_to_rgb(colors[i]) for i in range(len(colors))]
    # sns.lineplot(x_pct, means, linewidth=2.5, color=palette[i])

    plt.plot(x_pct, baseline, '--', marker='', color='black', linewidth=1.5, alpha=0.6, label="baseline")

    for i, type in enumerate(['SVM', 'fc', 'fine_tuning']):
        # set cleaner label names
        if type == 'SVM': label='SVM'
        if type == 'fc': label='FC'
        if type == 'fine_tuning': label='FT'

        # plot the mean values
        plt.errorbar(x_pct, results_dict[type]['means'], fmt='-o', color=rgb_colors[i], label=label)

        # plot the right error band
        if errortype == "minmax":
            plt.fill_between(x_pct, results_dict[type]['mins'], results_dict[type]['maxs'], color=rgb_colors[i], alpha=0.2)
        if errortype == "stds":
            plt.fill_between(x_pct, results_dict[type]['means']-results_dict[type]['stds'], results_dict[type]['means']+results_dict[type]['stds'], color=rgb_colors[i], alpha=0.2)
        if errortype == "ses":
            plt.fill_between(x_pct, results_dict[type]['means']-results_dict[type]['ses'], results_dict[type]['means']+results_dict[type]['ses'], color=rgb_colors[i], alpha=0.2)

    # set axis labels
    plt.xlabel("Fraction of images modified")
    if mode == 'skAUC': ylabel = 'AUC'
    if mode == 'acc': ylabel = 'Accuracy'
    plt.ylabel(ylabel)

    # set axes limits
    plt.xlim(0.05, 1.05)
    plt.ylim(0.45, 1.00)

    if 'small' in modification or 'imbalance' in modification:
        plt.xlim(1.05, 0.05)
        plt.xlabel("Fraction of dataset used")

    # set plot title and legend
    plt.title(plotname)
    plt.legend(loc="upper right")

    # save plot
    plt.savefig(os.path.join(plotpath, 'NOOUTL_individual_{}_{}_{}_{}.png'.format(dataset, mode, modification, errortype)))

    # plt.show()

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
                resultsfile = os.path.join(os.path.dirname(os.getcwd()), 'outputs/results/results_{}_{}_all.xlsx'.format(dataset, split))

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
        # modifications = ['small_easy', 'small_random', 'small_clusters',
        # 'small_hard', 'imbalance_classes', 'add_noise_speckle', 'add_noise_gaussian',
        # 'add_noise_poisson', 'add_noise_salt_and_pepper', 'grayscale',
        # 'hsv', 'image_rot', 'image_zoom', 'image_translation']

        modifications = ['small_easy', 'small_random', 'small_clusters',
        'small_hard', 'imbalance_classes', 'add_noise_gaussian', 'grayscale',
        'hsv', 'image_rot', 'image_zoom', 'image_translation']

        # modifications = ['small_easy', 'small_random', 'small_clusters', 'small_hard', 'imbalance_classes']

        # modifications = ['add_noise_gaussian', 'grayscale', 'hsv', 'image_rot', 'image_zoom', 'image_translation']

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
                    resultsfile = os.path.join(os.path.dirname(os.getcwd()), 'outputs/results/results_{}_{}_all.xlsx'.format(dataset, split))

                    # load the results
                    results = pd.read_excel(resultsfile, sheet_name=type)

                    # get the results for the modification
                    result_values = [results.query('source_dataset==@source_dataset')[mode].item() for source_dataset in results['source_dataset'] if mod in source_dataset]

                    # store values in a dictionary
                    results_dict[type][split] = result_values

                # create an anonymous function that gets every nth value for each split and makes a list out of it
                values = lambda n: [results_dict[type][split][n] for split in splits]

                # create a function that filters the values from the above values function
                # split values that are all 0.5 stay all 0.5, otherwise only keep the values that are not 0.5
                filter_values = lambda vals: list(filter(lambda a: a != 0.5, vals)) if (sum(vals) != 2.5) else vals

                for n in range(len(x_pct)):
                    # fractions for modifications that change the dataset size don't mean the same as for the other modifications
                    # for these, lower fractions correspond to more modification of the dataset
                    # correct for this
                    if mod in ['imbalance_classes', 'small_clusters', 'small_easy', 'small_hard', 'small_random']:
                        vals = values(len(x_pct)-1-n)

                        filtered_vals = filter_values(vals)

                        # add the mean values over every split for every fraction to dictionary
                        dict_df[str(x_pct[n])] = np.round(np.mean(filtered_vals),3)

                        # add difference between mean and minimum and maximum value to the corresponding dictionaries
                        min_values[str(x_pct[n])].append(np.round(np.mean(filtered_vals) - np.min(filtered_vals),3))
                        max_values[str(x_pct[n])].append(np.round(np.max(filtered_vals) - np.mean(filtered_vals),3))

                        # add standard deviation and standard error to the corresponding dictionaries
                        std_values[str(x_pct[n])].append(np.round(np.std(filtered_vals),3))
                        se_values[str(x_pct[n])].append(np.round(np.std(filtered_vals)/np.sqrt(len(filtered_vals)),3))


                        # # add the mean values over every split for every fraction to dictionary
                        # dict_df[str(x_pct[n])] = np.round(np.mean(values(len(x_pct)-1-n)),3)
                        #
                        # # add difference between mean and minimum and maximum value to the corresponding dictionaries
                        # min_values[str(x_pct[n])].append(np.round(np.mean(values(len(x_pct)-1-n)) - np.min(values(len(x_pct)-1-n)),3))
                        # max_values[str(x_pct[n])].append(np.round(np.max(values(len(x_pct)-1-n)) - np.mean(values(len(x_pct)-1-n)),3))
                        #
                        # # add standard deviation and standard error to the corresponding dictionaries
                        # std_values[str(x_pct[n])].append(np.round(np.std(values(len(x_pct)-1-n)),3))
                        # se_values[str(x_pct[n])].append(np.round(np.std(values(len(x_pct)-1-n))/np.sqrt(len(values(len(x_pct)-1-n))),3))
                    else:
                        vals = values(n)

                        filtered_vals = filter_values(vals)

                        # add the mean values over every split for every fraction to dictionary
                        dict_df[str(x_pct[n])] = np.round(np.mean(filtered_vals),3)

                        # add difference between mean and minimum and maximum value to the corresponding dictionaries
                        min_values[str(x_pct[n])].append(np.round(np.mean(filtered_vals) - np.min(filtered_vals),3))
                        max_values[str(x_pct[n])].append(np.round(np.max(filtered_vals) - np.mean(filtered_vals),3))

                        # add standard deviation and standard error to the corresponding dictionaries
                        std_values[str(x_pct[n])].append(np.round(np.std(filtered_vals),3))
                        se_values[str(x_pct[n])].append(np.round(np.std(filtered_vals)/np.sqrt(len(filtered_vals)),3))

                        # # add the mean values over every split for every fraction to dictionary
                        # dict_df[str(x_pct[n])] = np.round(np.mean(values(n)),3)
                        #
                        # # add difference between mean and minimum and maximum value to the corresponding dictionaries
                        # min_values[str(x_pct[n])].append(np.round(np.mean(values(n)) - np.min(values(n)),3))
                        # max_values[str(x_pct[n])].append(np.round(np.max(values(n)) - np.mean(values(n)),3))
                        #
                        # # add standard deviation and standard error to the corresponding dictionaries
                        # std_values[str(x_pct[n])].append(np.round(np.std(values(n)),3))
                        # se_values[str(x_pct[n])].append(np.round(np.std(values(n))/np.sqrt(len(values(n))),3))

                # add dictionary as a row to the dataframe
                data = data.append(dict_df, ignore_index=True)

        # get nice colors for plotting
        colors = ['#85cbcf', '#3984b6', '#1d2e81']

        # create anonymous function to convert hex values to rgb format
        hex_to_rgb = lambda hex: tuple(int(hex.lstrip('#')[i:i+2], 16) / 255.0 for i in (0, 2, 4))

        # convert the color palette to rgb
        rgb_colors = [hex_to_rgb(colors[i]) for i in range(len(colors))]

        # plot the mean points
        ax = sns.pointplot(y=frac_to_plot, x='modification', data=data, dodge=0.25, join=False, markers=['o', 's', '^'], scale=0.6, palette=rgb_colors, alpha=1.0, hue='type', ax=axs[f])

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
            ax.errorbar(rearranged_x_coords, rearranged_y_coords, yerr=[min_values[frac_to_plot], max_values[frac_to_plot]], ecolor=rgb_colors, fmt=' ')
        if errortype == "stds":
            ax.errorbar(rearranged_x_coords, rearranged_y_coords, yerr=std_values[frac_to_plot], ecolor=rgb_colors, fmt=' ', zorder=-1)
        if errortype == "ses":
            ax.errorbar(rearranged_x_coords, rearranged_y_coords, yerr=se_values[frac_to_plot], ecolor=rgb_colors, fmt=' ', zorder=-1)

        # only place legend on last subplot and in the right place
        if f != plots-1:
            axs[f].get_legend().remove()
        else:
            axs[f].legend(loc="lower right")

        # set y labels
        ylabels = ["Lowest $\delta$", "AUC\nMedium $\delta$", "Highest $\delta$"]
        for f in range(plots):
            axs[f].set_ylabel(ylabels[f])

    # hide x labels and tick labels for all but bottom plot and set grid lines
    for ax in axs:
        ax.label_outer()
        ax.grid(True, which='major', axis='y')

    # set cleaner labels for x axis
    # axs[-1].set_xticklabels(['A \n Size\neasy', 'B \n Size\nrandom', 'C \n Size\nclusters', 'D \n Size\nhard', 'E \n Class imbalance', 'F \n Speckle\nnoise', 'G \n Gaussian\nnoise', 'H \n Poisson\nnoise', 'I \n Salt & pepper\nnoise', 'J \n Grayscale', 'K \n Hue,\nSaturation,\nValue', 'L \n Rotation', 'M \n Zoom', 'N \n Translation'], fontweight='bold')
    axs[-1].set_xticklabels(['Size\neasy', 'Size\nrandom', 'Size\nclusters', 'Size\nhard', 'Class imbalance', 'Gaussian\nnoise', 'Grayscale', 'Hue,\nSaturation,\nValue', 'Rotation', 'Zoom', 'Translation'])
    # axs[-1].set_xticklabels(['Size\neasy', 'Size\nrandom', 'Size\nclusters', 'Size\nhard', 'Class imbalance'])
    # axs[-1].set_xticklabels(['Gaussian\nnoise', 'Grayscale', 'Hue,\nSaturation,\nValue', 'Rotation', 'Zoom', 'Translation'])


    plt.xlabel('')
    # save plot
    # plt.savefig(os.path.join(plotpath, '{}_{}_{}_grouped-3-2.png'.format(dataset, mode, errortype)))
    plt.show()
