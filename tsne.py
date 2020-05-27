from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.models import load_model
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
from sklearn.manifold import TSNE
from PIL import Image
import rasterfairy
from main import NeuralNetwork
import seaborn as sns


def main():
    """Perform t-SNE."""
    # read parameters for wanted dataset from config file
    with open('config.json') as json_file:
        config_json = json.load(json_file)
        config = config_json[args['dataset']]

    # create directories to save results if they don't exist yet
    resultspath = os.path.join(os.path.dirname(os.getcwd()), 'results')
    if not os.path.exists(resultspath):
        os.makedirs(resultspath)

    # set a random seed
    seed = 28

    # if features are not yet extracted, extract them
    if not os.path.exists(os.path.join(config['output_path'], 'pca_features_50.p')):
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
        print("extracting features...")
        bn_features_train, bn_features_test, true_labels_train, true_labels_test = network.extract_bottleneck_features()

        # scale the data to zero mean, unit variance for PCA
        scaler = StandardScaler()
        train_features = scaler.fit_transform(bn_features_train)

        # fit PCA
        print("applying PCA...")
        # pca = PCA(.90)
        pca = PCA(n_components=50)
        # print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca.explained_variance_ratio_)))
        pca.fit(train_features)

        # apply PCA to features and test data
        reduced_train_features = pca.transform(train_features)

        # create list of full file paths
        train_paths = [os.path.join(config['trainingpath'], filename) for filename in network.gen_training.filenames]

        # convert to arrays
        pca_features = np.array(reduced_train_features)
        images = np.array(train_paths)

        # save extracted features in a pickle file
        print("saving features...")
        pickle.dump([images, pca_features, pca], open(os.path.join(config['output_path'], 'pca_features_50.p'), 'wb'))

    else:
        # load features if they are already once extracted
        print("loading features...")
        images, pca_features, pca = pickle.load(open(os.path.join(config['output_path'], 'pca_features_50.p'), 'rb'))

    for img, f in list(zip(images, pca_features))[0:5]:
        print("image: %s, features: %0.2f,%0.2f,%0.2f,%0.2f... " % (img, f[0], f[1], f[2], f[3]))

    # only plot the input amount of images, if no input is given or input is larger than number of images, use all images
    if args['num_images'] == 0 or args['num_images'] > len(images):
        num_images_to_plot = len(images)
    else:
        num_images_to_plot = args['num_images']

    print("number of images used for t-SNE: {}".format(num_images_to_plot))

    # set random seed to always get the same random samples
    random.seed(seed)
    if len(images) > num_images_to_plot:
        sort_order = sorted(random.sample(range(len(images)), num_images_to_plot))
        images = [images[i] for i in sort_order]
        pca_features = [pca_features[i] for i in sort_order]

    # # only take the number of images to plot
    # images = images[:num_images_to_plot]
    # pca_features = pca_features[:num_images_to_plot]

    print("performing t-SNE...")
    if args['dims'] == '2D':
        perplexity = int(np.sqrt(num_images_to_plot))
        tsne = TSNE(n_components=2, learning_rate=150, perplexity=perplexity, random_state=seed, angle=0.2, verbose=2).fit_transform(np.array(pca_features))
    if args['dims'] == '3D':
        perplexity = int(np.sqrt(num_images_to_plot))
        tsne = TSNE(n_components=3, learning_rate=150, perplexity=perplexity, random_state=seed, angle=0.2, verbose=2).fit_transform(np.array(pca_features))

    if args['mode'] == 'images':
        print("creating t-SNE image...")
        # normalize the embedding
        tx, ty = tsne[:, 0], tsne[:, 1]
        tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

        width = 4000
        height = 3000
        max_dim = 100

        full_image = Image.new('RGBA', (width, height))
        for img, x, y in zip(images, tx, ty):
            tile = Image.open(img)
            if 'ISIC' in args['dataset']:
                if 'malignant' in img:
                    old_size = tile.size
                    new_size = (old_size[0]+20, old_size[1]+20)
                    new_im = Image.new("RGB", new_size, "red")

                    new_im.paste(tile, (int((new_size[0]-old_size[0])/2), int((new_size[1]-old_size[1])/2)))
                    tile = new_im
                if 'benign' in img:
                    old_size = tile.size
                    new_size = (old_size[0]+20, old_size[1]+20)
                    new_im = Image.new("RGB", new_size, "green")

                    new_im.paste(tile, (int((new_size[0]-old_size[0])/2), int((new_size[1]-old_size[1])/2)))
                    tile = new_im

            if 'CNMC' in args['dataset']:
                if 'leukemic' in img:
                    old_size = tile.size
                    new_size = (old_size[0]+20, old_size[1]+20)
                    new_im = Image.new("RGB", new_size, "red")

                    new_im.paste(tile, (int((new_size[0]-old_size[0])/2), int((new_size[1]-old_size[1])/2)))
                    tile = new_im
                if 'normal' in img:
                    old_size = tile.size
                    new_size = (old_size[0]+20, old_size[1]+20)
                    new_im = Image.new("RGB", new_size, "green")

                    new_im.paste(tile, (int((new_size[0]-old_size[0])/2), int((new_size[1]-old_size[1])/2)))
                    tile = new_im

            rs = max(1, tile.width/max_dim, tile.height/max_dim)
            tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
            full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

        plt.figure(figsize=(16, 12))
        plt.imshow(full_image)

        full_image.save(os.path.join(config['output_path'], 'tSNE-images-{}-{}-2D-pca_50.png'.format(args['dataset'], num_images_to_plot)))

        print("creating t-SNE grid image...")
        # get dimensions for the raster that is closest to a square, where all the images fit
        max_side = int(num_images_to_plot**(1/2.0))
        for ny in range(2, max_side+1)[::-1]:
            nx = num_images_to_plot // ny
            if (ny * nx) == num_images_to_plot:
                break

        # assign to grid
        grid_assignment = rasterfairy.transformPointCloud2D(tsne, target=(nx, ny))

        tile_width = 70
        tile_height = 70

        full_width = tile_width * nx
        full_height = tile_height * ny
        aspect_ratio = float(tile_width) / tile_height

        grid_image = Image.new('RGB', (full_width, full_height))

        for img, grid_pos in zip(images, grid_assignment[0]):
            idx_x, idx_y = grid_pos
            x, y = tile_width * idx_x, tile_height * idx_y
            # tile = Image.open(img)
            tile = Image.open(img)
            if 'ISIC' in args['dataset']:
                if 'malignant' in img:
                    old_size = tile.size
                    new_size = (old_size[0]+20, old_size[1]+20)
                    new_im = Image.new("RGB", new_size, "red")
                    new_im.paste(tile, (int((new_size[0]-old_size[0])/2), int((new_size[1]-old_size[1])/2)))
                    tile = new_im
                if 'benign' in img:
                    old_size = tile.size
                    new_size = (old_size[0]+20, old_size[1]+20)
                    new_im = Image.new("RGB", new_size, "green")
                    new_im.paste(tile, (int((new_size[0]-old_size[0])/2), int((new_size[1]-old_size[1])/2)))
                    tile = new_im

            if 'CNMC' in args['dataset']:
                if 'leukemic' in img:
                    old_size = tile.size
                    new_size = (old_size[0]+20, old_size[1]+20)
                    new_im = Image.new("RGB", new_size, "red")
                    new_im.paste(tile, (int((new_size[0]-old_size[0])/2), int((new_size[1]-old_size[1])/2)))
                    tile = new_im
                if 'normal' in img:
                    old_size = tile.size
                    new_size = (old_size[0]+20, old_size[1]+20)
                    new_im = Image.new("RGB", new_size, "green")
                    new_im.paste(tile, (int((new_size[0]-old_size[0])/2), int((new_size[1]-old_size[1])/2)))
                    tile = new_im

            tile_ar = float(tile.width) / tile.height  # center-crop the tile to match aspect_ratio
            if (tile_ar > aspect_ratio):
                margin = 0.5 * (tile.width - aspect_ratio * tile.height)
                tile = tile.crop((margin, 0, margin + aspect_ratio * tile.height, tile.height))
            else:
                margin = 0.5 * (tile.height - float(tile.width) / aspect_ratio)
                tile = tile.crop((0, margin, tile.width, margin + float(tile.width) / aspect_ratio))
            tile = tile.resize((tile_width, tile_height), Image.ANTIALIAS)
            grid_image.paste(tile, (int(x), int(y)))

        plt.figure(figsize=(16, 12))
        plt.imshow(grid_image)

        grid_image.save(os.path.join(config['output_path'], 'tSNE-grid-{}-{}-pca_50.png'.format(args['dataset'], num_images_to_plot)))

    # for plotting points
    if args['mode'] == 'points':
        if args['dims'] == '2D':
            print("creating t-SNE image...")

            plt.figure(figsize=(16, 10))

            tx, ty = tsne[:, 0], tsne[:, 1]
            y = []

            # create labels for color coding
            for img in images:
                if 'ISIC' in args['dataset']:
                    if 'malignant' in img:
                        y.append(0)
                    if 'benign' in img:
                        y.append(1)

                if 'CNMC' in args['dataset']:
                    if 'leukemic' in img:
                        y.append(0)
                    if 'normal' in img:
                        y.append(1)

            # set red and green color palette for the classes
            palette = sns.color_palette(['#00FF00', '#FF0000'])

            # plot the data
            sns.scatterplot(tx, ty, hue=y, legend='full', palette=palette)

            # set legend
            if 'ISIC' in args['dataset']:
                plt.legend(labels=['malignant', 'benign'], loc="upper right")
            if 'CNMC' in args['dataset']:
                plt.legend(labels=['leukemic', 'normal'], loc="upper right")

            # save the t-SNE plot
            plt.savefig(os.path.join(config['output_path'], 'tSNE-points-{}-{}-2D-pca_50.png'.format(args['dataset'], num_images_to_plot)))

        if args['dims'] == '3D':
            print("creating t-SNE image...")

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            tx, ty, tz = tsne[:, 0], tsne[:, 1], tsne[:, 2]
            y = []

            # create labels for color coding
            for img in images:
                if 'ISIC' in args['dataset']:
                    if 'malignant' in img:
                        y.append(0)
                    if 'benign' in img:
                        y.append(1)

                if 'CNMC' in args['dataset']:
                    if 'leukemic' in img:
                        y.append(0)
                    if 'normal' in img:
                        y.append(1)

            ax.scatter(tx, ty, tz, c=y, cmap="RdYlGn_r", alpha=1.0)

            # set legend
            if 'ISIC' in args['dataset']:
                plt.legend(labels=['benign', 'malignant'], loc="upper right")
            if 'CNMC' in args['dataset']:
                plt.legend(labels=['normal', 'leukemic'], loc="upper right")

            # save the t-SNE plot
            plt.savefig(os.path.join(config['output_path'], 'tSNE-points-{}-{}-3D-pca_50.png'.format(args['dataset'], num_images_to_plot)))


if __name__ == "__main__":
    # choose GPU for training
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    # construct argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
        '--dataset',
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
                'ISIC_2_imbalance_classes_f=0.9', 'ISIC_2_imbalance_classes_f=1.0', 'ISIC_2_grayscale_f=0.1',
                'ISIC_2_grayscale_f=0.2', 'ISIC_2_grayscale_f=0.3', 'ISIC_2_grayscale_f=0.4',
                'ISIC_2_grayscale_f=0.5', 'ISIC_2_grayscale_f=0.6', 'ISIC_2_grayscale_f=0.7',
                'ISIC_2_grayscale_f=0.8', 'ISIC_2_grayscale_f=0.9', 'ISIC_2_grayscale_f=1.0',
                'ISIC_2_hsv_f=0.1', 'ISIC_2_hsv_f=0.2', 'ISIC_2_hsv_f=0.3', 'ISIC_2_hsv_f=0.4',
                'ISIC_2_hsv_f=0.5', 'ISIC_2_hsv_f=0.6', 'ISIC_2_hsv_f=0.7',
                'ISIC_2_hsv_f=0.8', 'ISIC_2_hsv_f=0.9', 'ISIC_2_hsv_f=1.0',
                'ISIC_2_blur_f=0.1', 'ISIC_2_blur_f=0.2', 'ISIC_2_blur_f=0.3', 'ISIC_2_blur_f=0.4',
                'ISIC_2_blur_f=0.5', 'ISIC_2_blur_f=0.6', 'ISIC_2_blur_f=0.7',
                'ISIC_2_blur_f=0.8', 'ISIC_2_blur_f=0.9', 'ISIC_2_blur_f=1.0',
                'ISIC_2_small_random_f=0.1', 'ISIC_2_small_random_f=0.2', 'ISIC_2_small_random_f=0.3', 'ISIC_2_small_random_f=0.4',
                'ISIC_2_small_random_f=0.5', 'ISIC_2_small_random_f=0.6', 'ISIC_2_small_random_f=0.7',
                'ISIC_2_small_random_f=0.8', 'ISIC_2_small_random_f=0.9', 'ISIC_2_small_random_f=1.0',
                'ISIC_2_small_easy_f=0.1', 'ISIC_2_small_easy_f=0.2', 'ISIC_2_small_easy_f=0.3', 'ISIC_2_small_easy_f=0.4',
                'ISIC_2_small_easy_f=0.5', 'ISIC_2_small_easy_f=0.6', 'ISIC_2_small_easy_f=0.7',
                'ISIC_2_small_easy_f=0.8', 'ISIC_2_small_easy_f=0.9', 'ISIC_2_small_easy_f=1.0',
                'ISIC_2_small_hard_f=0.1', 'ISIC_2_small_hard_f=0.2', 'ISIC_2_small_hard_f=0.3', 'ISIC_2_small_hard_f=0.4',
                'ISIC_2_small_hard_f=0.5', 'ISIC_2_small_hard_f=0.6', 'ISIC_2_small_hard_f=0.7',
                'ISIC_2_small_hard_f=0.8', 'ISIC_2_small_hard_f=0.9', 'ISIC_2_small_hard_f=1.0',
                'ISIC_2_small_clusters_f=0.1', 'ISIC_2_small_clusters_f=0.2', 'ISIC_2_small_clusters_f=0.3', 'ISIC_2_small_clusters_f=0.4',
                'ISIC_2_small_clusters_f=0.5', 'ISIC_2_small_clusters_f=0.6', 'ISIC_2_small_clusters_f=0.7',
                'ISIC_2_small_clusters_f=0.8', 'ISIC_2_small_clusters_f=0.9', 'ISIC_2_small_clusters_f=1.0',
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
                'ISIC_3_imbalance_classes_f=0.9', 'ISIC_3_imbalance_classes_f=1.0', 'ISIC_3_grayscale_f=0.1',
                'ISIC_3_grayscale_f=0.2', 'ISIC_3_grayscale_f=0.3', 'ISIC_3_grayscale_f=0.4',
                'ISIC_3_grayscale_f=0.5', 'ISIC_3_grayscale_f=0.6', 'ISIC_3_grayscale_f=0.7',
                'ISIC_3_grayscale_f=0.8', 'ISIC_3_grayscale_f=0.9', 'ISIC_3_grayscale_f=1.0',
                'ISIC_3_hsv_f=0.1', 'ISIC_3_hsv_f=0.2', 'ISIC_3_hsv_f=0.3', 'ISIC_3_hsv_f=0.4',
                'ISIC_3_hsv_f=0.5', 'ISIC_3_hsv_f=0.6', 'ISIC_3_hsv_f=0.7',
                'ISIC_3_hsv_f=0.8', 'ISIC_3_hsv_f=0.9', 'ISIC_3_hsv_f=1.0',
                'ISIC_3_blur_f=0.1', 'ISIC_3_blur_f=0.2', 'ISIC_3_blur_f=0.3', 'ISIC_3_blur_f=0.4',
                'ISIC_3_blur_f=0.5', 'ISIC_3_blur_f=0.6', 'ISIC_3_blur_f=0.7',
                'ISIC_3_blur_f=0.8', 'ISIC_3_blur_f=0.9', 'ISIC_3_blur_f=1.0',
                'ISIC_3_small_random_f=0.1', 'ISIC_3_small_random_f=0.2', 'ISIC_3_small_random_f=0.3', 'ISIC_3_small_random_f=0.4',
                'ISIC_3_small_random_f=0.5', 'ISIC_3_small_random_f=0.6', 'ISIC_3_small_random_f=0.7',
                'ISIC_3_small_random_f=0.8', 'ISIC_3_small_random_f=0.9', 'ISIC_3_small_random_f=1.0',
                'ISIC_3_small_easy_f=0.1', 'ISIC_3_small_easy_f=0.2', 'ISIC_3_small_easy_f=0.3', 'ISIC_3_small_easy_f=0.4',
                'ISIC_3_small_easy_f=0.5', 'ISIC_3_small_easy_f=0.6', 'ISIC_3_small_easy_f=0.7',
                'ISIC_3_small_easy_f=0.8', 'ISIC_3_small_easy_f=0.9', 'ISIC_3_small_easy_f=1.0',
                'ISIC_3_small_hard_f=0.1', 'ISIC_3_small_hard_f=0.2', 'ISIC_3_small_hard_f=0.3', 'ISIC_3_small_hard_f=0.4',
                'ISIC_3_small_hard_f=0.5', 'ISIC_3_small_hard_f=0.6', 'ISIC_3_small_hard_f=0.7',
                'ISIC_3_small_hard_f=0.8', 'ISIC_3_small_hard_f=0.9', 'ISIC_3_small_hard_f=1.0',
                'ISIC_3_small_clusters_f=0.1', 'ISIC_3_small_clusters_f=0.2', 'ISIC_3_small_clusters_f=0.3', 'ISIC_3_small_clusters_f=0.4',
                'ISIC_3_small_clusters_f=0.5', 'ISIC_3_small_clusters_f=0.6', 'ISIC_3_small_clusters_f=0.7',
                'ISIC_3_small_clusters_f=0.8', 'ISIC_3_small_clusters_f=0.9', 'ISIC_3_small_clusters_f=1.0',
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
                'ISIC_4_imbalance_classes_f=0.9', 'ISIC_4_imbalance_classes_f=1.0', 'ISIC_4_grayscale_f=0.1',
                'ISIC_4_grayscale_f=0.2', 'ISIC_4_grayscale_f=0.3', 'ISIC_4_grayscale_f=0.4',
                'ISIC_4_grayscale_f=0.5', 'ISIC_4_grayscale_f=0.6', 'ISIC_4_grayscale_f=0.7',
                'ISIC_4_grayscale_f=0.8', 'ISIC_4_grayscale_f=0.9', 'ISIC_4_grayscale_f=1.0',
                'ISIC_4_hsv_f=0.1', 'ISIC_4_hsv_f=0.2', 'ISIC_4_hsv_f=0.3', 'ISIC_4_hsv_f=0.4',
                'ISIC_4_hsv_f=0.5', 'ISIC_4_hsv_f=0.6', 'ISIC_4_hsv_f=0.7',
                'ISIC_4_hsv_f=0.8', 'ISIC_4_hsv_f=0.9', 'ISIC_4_hsv_f=1.0',
                'ISIC_4_blur_f=0.1', 'ISIC_4_blur_f=0.2', 'ISIC_4_blur_f=0.3', 'ISIC_4_blur_f=0.4',
                'ISIC_4_blur_f=0.5', 'ISIC_4_blur_f=0.6', 'ISIC_4_blur_f=0.7',
                'ISIC_4_blur_f=0.8', 'ISIC_4_blur_f=0.9', 'ISIC_4_blur_f=1.0',
                'ISIC_4_small_random_f=0.1', 'ISIC_4_small_random_f=0.2', 'ISIC_4_small_random_f=0.3', 'ISIC_4_small_random_f=0.4',
                'ISIC_4_small_random_f=0.5', 'ISIC_4_small_random_f=0.6', 'ISIC_4_small_random_f=0.7',
                'ISIC_4_small_random_f=0.8', 'ISIC_4_small_random_f=0.9', 'ISIC_4_small_random_f=1.0',
                'ISIC_4_small_easy_f=0.1', 'ISIC_4_small_easy_f=0.2', 'ISIC_4_small_easy_f=0.3', 'ISIC_4_small_easy_f=0.4',
                'ISIC_4_small_easy_f=0.5', 'ISIC_4_small_easy_f=0.6', 'ISIC_4_small_easy_f=0.7',
                'ISIC_4_small_easy_f=0.8', 'ISIC_4_small_easy_f=0.9', 'ISIC_4_small_easy_f=1.0',
                'ISIC_4_small_hard_f=0.1', 'ISIC_4_small_hard_f=0.2', 'ISIC_4_small_hard_f=0.3', 'ISIC_4_small_hard_f=0.4',
                'ISIC_4_small_hard_f=0.5', 'ISIC_4_small_hard_f=0.6', 'ISIC_4_small_hard_f=0.7',
                'ISIC_4_small_hard_f=0.8', 'ISIC_4_small_hard_f=0.9', 'ISIC_4_small_hard_f=1.0',
                'ISIC_4_small_clusters_f=0.1', 'ISIC_4_small_clusters_f=0.2', 'ISIC_4_small_clusters_f=0.3', 'ISIC_4_small_clusters_f=0.4',
                'ISIC_4_small_clusters_f=0.5', 'ISIC_4_small_clusters_f=0.6', 'ISIC_4_small_clusters_f=0.7',
                'ISIC_4_small_clusters_f=0.8', 'ISIC_4_small_clusters_f=0.9', 'ISIC_4_small_clusters_f=1.0',
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
                'ISIC_5_imbalance_classes_f=0.9', 'ISIC_5_imbalance_classes_f=1.0', 'ISIC_5_grayscale_f=0.1',
                'ISIC_5_grayscale_f=0.2', 'ISIC_5_grayscale_f=0.3', 'ISIC_5_grayscale_f=0.4',
                'ISIC_5_grayscale_f=0.5', 'ISIC_5_grayscale_f=0.6', 'ISIC_5_grayscale_f=0.7',
                'ISIC_5_grayscale_f=0.8', 'ISIC_5_grayscale_f=0.9', 'ISIC_5_grayscale_f=1.0',
                'ISIC_5_hsv_f=0.1', 'ISIC_5_hsv_f=0.2', 'ISIC_5_hsv_f=0.3', 'ISIC_5_hsv_f=0.4',
                'ISIC_5_hsv_f=0.5', 'ISIC_5_hsv_f=0.6', 'ISIC_5_hsv_f=0.7',
                'ISIC_5_hsv_f=0.8', 'ISIC_5_hsv_f=0.9', 'ISIC_5_hsv_f=1.0',
                'ISIC_5_blur_f=0.1', 'ISIC_5_blur_f=0.2', 'ISIC_5_blur_f=0.3', 'ISIC_5_blur_f=0.4',
                'ISIC_5_blur_f=0.5', 'ISIC_5_blur_f=0.6', 'ISIC_5_blur_f=0.7',
                'ISIC_5_blur_f=0.8', 'ISIC_5_blur_f=0.9', 'ISIC_5_blur_f=1.0',
                'ISIC_5_small_random_f=0.1', 'ISIC_5_small_random_f=0.2', 'ISIC_5_small_random_f=0.3', 'ISIC_5_small_random_f=0.4',
                'ISIC_5_small_random_f=0.5', 'ISIC_5_small_random_f=0.6', 'ISIC_5_small_random_f=0.7',
                'ISIC_5_small_random_f=0.8', 'ISIC_5_small_random_f=0.9', 'ISIC_5_small_random_f=1.0',
                'ISIC_5_small_easy_f=0.1', 'ISIC_5_small_easy_f=0.2', 'ISIC_5_small_easy_f=0.3', 'ISIC_5_small_easy_f=0.4',
                'ISIC_5_small_easy_f=0.5', 'ISIC_5_small_easy_f=0.6', 'ISIC_5_small_easy_f=0.7',
                'ISIC_5_small_easy_f=0.8', 'ISIC_5_small_easy_f=0.9', 'ISIC_5_small_easy_f=1.0',
                'ISIC_5_small_hard_f=0.1', 'ISIC_5_small_hard_f=0.2', 'ISIC_5_small_hard_f=0.3', 'ISIC_5_small_hard_f=0.4',
                'ISIC_5_small_hard_f=0.5', 'ISIC_5_small_hard_f=0.6', 'ISIC_5_small_hard_f=0.7',
                'ISIC_5_small_hard_f=0.8', 'ISIC_5_small_hard_f=0.9', 'ISIC_5_small_hard_f=1.0',
                'ISIC_5_small_clusters_f=0.1', 'ISIC_5_small_clusters_f=0.2', 'ISIC_5_small_clusters_f=0.3', 'ISIC_5_small_clusters_f=0.4',
                'ISIC_5_small_clusters_f=0.5', 'ISIC_5_small_clusters_f=0.6', 'ISIC_5_small_clusters_f=0.7',
                'ISIC_5_small_clusters_f=0.8', 'ISIC_5_small_clusters_f=0.9', 'ISIC_5_small_clusters_f=1.0',
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
                'ISIC_6_imbalance_classes_f=0.9', 'ISIC_6_imbalance_classes_f=1.0', 'ISIC_6_grayscale_f=0.1',
                'ISIC_6_grayscale_f=0.2', 'ISIC_6_grayscale_f=0.3', 'ISIC_6_grayscale_f=0.4',
                'ISIC_6_grayscale_f=0.5', 'ISIC_6_grayscale_f=0.6', 'ISIC_6_grayscale_f=0.7',
                'ISIC_6_grayscale_f=0.8', 'ISIC_6_grayscale_f=0.9', 'ISIC_6_grayscale_f=1.0',
                'ISIC_6_hsv_f=0.1', 'ISIC_6_hsv_f=0.2', 'ISIC_6_hsv_f=0.3', 'ISIC_6_hsv_f=0.4',
                'ISIC_6_hsv_f=0.5', 'ISIC_6_hsv_f=0.6', 'ISIC_6_hsv_f=0.7',
                'ISIC_6_hsv_f=0.8', 'ISIC_6_hsv_f=0.9', 'ISIC_6_hsv_f=1.0',
                'ISIC_6_blur_f=0.1', 'ISIC_6_blur_f=0.2', 'ISIC_6_blur_f=0.3', 'ISIC_6_blur_f=0.4',
                'ISIC_6_blur_f=0.5', 'ISIC_6_blur_f=0.6', 'ISIC_6_blur_f=0.7',
                'ISIC_6_blur_f=0.8', 'ISIC_6_blur_f=0.9', 'ISIC_6_blur_f=1.0',
                'ISIC_6_small_random_f=0.1', 'ISIC_6_small_random_f=0.2', 'ISIC_6_small_random_f=0.3', 'ISIC_6_small_random_f=0.4',
                'ISIC_6_small_random_f=0.5', 'ISIC_6_small_random_f=0.6', 'ISIC_6_small_random_f=0.7',
                'ISIC_6_small_random_f=0.8', 'ISIC_6_small_random_f=0.9', 'ISIC_6_small_random_f=1.0',
                'ISIC_6_small_easy_f=0.1', 'ISIC_6_small_easy_f=0.2', 'ISIC_6_small_easy_f=0.3', 'ISIC_6_small_easy_f=0.4',
                'ISIC_6_small_easy_f=0.5', 'ISIC_6_small_easy_f=0.6', 'ISIC_6_small_easy_f=0.7',
                'ISIC_6_small_easy_f=0.8', 'ISIC_6_small_easy_f=0.9', 'ISIC_6_small_easy_f=1.0',
                'ISIC_6_small_hard_f=0.1', 'ISIC_6_small_hard_f=0.2', 'ISIC_6_small_hard_f=0.3', 'ISIC_6_small_hard_f=0.4',
                'ISIC_6_small_hard_f=0.5', 'ISIC_6_small_hard_f=0.6', 'ISIC_6_small_hard_f=0.7',
                'ISIC_6_small_hard_f=0.8', 'ISIC_6_small_hard_f=0.9', 'ISIC_6_small_hard_f=1.0',
                'ISIC_6_small_clusters_f=0.1', 'ISIC_6_small_clusters_f=0.2', 'ISIC_6_small_clusters_f=0.3', 'ISIC_6_small_clusters_f=0.4',
                'ISIC_6_small_clusters_f=0.5', 'ISIC_6_small_clusters_f=0.6', 'ISIC_6_small_clusters_f=0.7',
                'ISIC_6_small_clusters_f=0.8', 'ISIC_6_small_clusters_f=0.9', 'ISIC_6_small_clusters_f=1.0',
                'CNMC_2', 'CNMC_2_image_rot_f=0.1', 'CNMC_2_image_rot_f=0.2',
                'CNMC_2_image_rot_f=0.3', 'CNMC_2_image_rot_f=0.4', 'CNMC_2_image_rot_f=0.5',
                'CNMC_2_image_rot_f=0.6', 'CNMC_2_image_rot_f=0.7', 'CNMC_2_image_rot_f=0.8',
                'CNMC_2_image_rot_f=0.9', 'CNMC_2_image_rot_f=1.0', 'CNMC_2_image_translation_f=0.1',
                'CNMC_2_image_translation_f=0.2', 'CNMC_2_image_translation_f=0.3', 'CNMC_2_image_translation_f=0.4',
                'CNMC_2_image_translation_f=0.5', 'CNMC_2_image_translation_f=0.6', 'CNMC_2_image_translation_f=0.7',
                'CNMC_2_image_translation_f=0.8', 'CNMC_2_image_translation_f=0.9', 'CNMC_2_image_translation_f=1.0',
                'CNMC_2_image_zoom_f=0.1', 'CNMC_2_image_zoom_f=0.2', 'CNMC_2_image_zoom_f=0.3',
                'CNMC_2_image_zoom_f=0.4', 'CNMC_2_image_zoom_f=0.5', 'CNMC_2_image_zoom_f=0.6',
                'CNMC_2_image_zoom_f=0.7', 'CNMC_2_image_zoom_f=0.8', 'CNMC_2_image_zoom_f=0.9',
                'CNMC_2_image_zoom_f=1.0', 'CNMC_2_add_noise_gaussian_f=0.1', 'CNMC_2_add_noise_gaussian_f=0.2',
                'CNMC_2_add_noise_gaussian_f=0.3', 'CNMC_2_add_noise_gaussian_f=0.4', 'CNMC_2_add_noise_gaussian_f=0.5',
                'CNMC_2_add_noise_gaussian_f=0.6', 'CNMC_2_add_noise_gaussian_f=0.7', 'CNMC_2_add_noise_gaussian_f=0.8',
                'CNMC_2_add_noise_gaussian_f=0.9', 'CNMC_2_add_noise_gaussian_f=1.0', 'CNMC_2_add_noise_poisson_f=0.1',
                'CNMC_2_add_noise_poisson_f=0.2', 'CNMC_2_add_noise_poisson_f=0.3', 'CNMC_2_add_noise_poisson_f=0.4',
                'CNMC_2_add_noise_poisson_f=0.5', 'CNMC_2_add_noise_poisson_f=0.6', 'CNMC_2_add_noise_poisson_f=0.7',
                'CNMC_2_add_noise_poisson_f=0.8', 'CNMC_2_add_noise_poisson_f=0.9', 'CNMC_2_add_noise_poisson_f=1.0',
                'CNMC_2_add_noise_salt_and_pepper_f=0.1', 'CNMC_2_add_noise_salt_and_pepper_f=0.2',
                'CNMC_2_add_noise_salt_and_pepper_f=0.3', 'CNMC_2_add_noise_salt_and_pepper_f=0.4',
                'CNMC_2_add_noise_salt_and_pepper_f=0.5', 'CNMC_2_add_noise_salt_and_pepper_f=0.6',
                'CNMC_2_add_noise_salt_and_pepper_f=0.7', 'CNMC_2_add_noise_salt_and_pepper_f=0.8',
                'CNMC_2_add_noise_salt_and_pepper_f=0.9', 'CNMC_2_add_noise_salt_and_pepper_f=1.0',
                'CNMC_2_add_noise_speckle_f=0.1', 'CNMC_2_add_noise_speckle_f=0.2', 'CNMC_2_add_noise_speckle_f=0.3',
                'CNMC_2_add_noise_speckle_f=0.4', 'CNMC_2_add_noise_speckle_f=0.5', 'CNMC_2_add_noise_speckle_f=0.6',
                'CNMC_2_add_noise_speckle_f=0.7', 'CNMC_2_add_noise_speckle_f=0.8', 'CNMC_2_add_noise_speckle_f=0.9',
                'CNMC_2_add_noise_speckle_f=1.0', 'CNMC_2_imbalance_classes_f=0.1', 'CNMC_2_imbalance_classes_f=0.2',
                'CNMC_2_imbalance_classes_f=0.3', 'CNMC_2_imbalance_classes_f=0.4', 'CNMC_2_imbalance_classes_f=0.5',
                'CNMC_2_imbalance_classes_f=0.6', 'CNMC_2_imbalance_classes_f=0.7', 'CNMC_2_imbalance_classes_f=0.8',
                'CNMC_2_imbalance_classes_f=0.9', 'CNMC_2_imbalance_classes_f=1.0', 'CNMC_2_grayscale_f=0.1',
                'CNMC_2_grayscale_f=0.2', 'CNMC_2_grayscale_f=0.3', 'CNMC_2_grayscale_f=0.4',
                'CNMC_2_grayscale_f=0.5', 'CNMC_2_grayscale_f=0.6', 'CNMC_2_grayscale_f=0.7',
                'CNMC_2_grayscale_f=0.8', 'CNMC_2_grayscale_f=0.9', 'CNMC_2_grayscale_f=1.0',
                'CNMC_2_hsv_f=0.1', 'CNMC_2_hsv_f=0.2', 'CNMC_2_hsv_f=0.3', 'CNMC_2_hsv_f=0.4',
                'CNMC_2_hsv_f=0.5', 'CNMC_2_hsv_f=0.6', 'CNMC_2_hsv_f=0.7',
                'CNMC_2_hsv_f=0.8', 'CNMC_2_hsv_f=0.9', 'CNMC_2_hsv_f=1.0',
                'CNMC_2_blur_f=0.1', 'CNMC_2_blur_f=0.2', 'CNMC_2_blur_f=0.3', 'CNMC_2_blur_f=0.4',
                'CNMC_2_blur_f=0.5', 'CNMC_2_blur_f=0.6', 'CNMC_2_blur_f=0.7',
                'CNMC_2_blur_f=0.8', 'CNMC_2_blur_f=0.9', 'CNMC_2_blur_f=1.0',
                'CNMC_2_small_random_f=0.1', 'CNMC_2_small_random_f=0.2', 'CNMC_2_small_random_f=0.3', 'CNMC_2_small_random_f=0.4',
                'CNMC_2_small_random_f=0.5', 'CNMC_2_small_random_f=0.6', 'CNMC_2_small_random_f=0.7',
                'CNMC_2_small_random_f=0.8', 'CNMC_2_small_random_f=0.9', 'CNMC_2_small_random_f=1.0',
                'CNMC_2_small_easy_f=0.1', 'CNMC_2_small_easy_f=0.2', 'CNMC_2_small_easy_f=0.3', 'CNMC_2_small_easy_f=0.4',
                'CNMC_2_small_easy_f=0.5', 'CNMC_2_small_easy_f=0.6', 'CNMC_2_small_easy_f=0.7',
                'CNMC_2_small_easy_f=0.8', 'CNMC_2_small_easy_f=0.9', 'CNMC_2_small_easy_f=1.0',
                'CNMC_2_small_hard_f=0.1', 'CNMC_2_small_hard_f=0.2', 'CNMC_2_small_hard_f=0.3', 'CNMC_2_small_hard_f=0.4',
                'CNMC_2_small_hard_f=0.5', 'CNMC_2_small_hard_f=0.6', 'CNMC_2_small_hard_f=0.7',
                'CNMC_2_small_hard_f=0.8', 'CNMC_2_small_hard_f=0.9', 'CNMC_2_small_hard_f=1.0',
                'CNMC_2_small_clusters_f=0.1', 'CNMC_2_small_clusters_f=0.2', 'CNMC_2_small_clusters_f=0.3', 'CNMC_2_small_clusters_f=0.4',
                'CNMC_2_small_clusters_f=0.5', 'CNMC_2_small_clusters_f=0.6', 'CNMC_2_small_clusters_f=0.7',
                'CNMC_2_small_clusters_f=0.8', 'CNMC_2_small_clusters_f=0.9', 'CNMC_2_small_clusters_f=1.0',
                'CNMC_3', 'CNMC_3_image_rot_f=0.1', 'CNMC_3_image_rot_f=0.2',
                'CNMC_3_image_rot_f=0.3', 'CNMC_3_image_rot_f=0.4', 'CNMC_3_image_rot_f=0.5',
                'CNMC_3_image_rot_f=0.6', 'CNMC_3_image_rot_f=0.7', 'CNMC_3_image_rot_f=0.8',
                'CNMC_3_image_rot_f=0.9', 'CNMC_3_image_rot_f=1.0', 'CNMC_3_image_translation_f=0.1',
                'CNMC_3_image_translation_f=0.2', 'CNMC_3_image_translation_f=0.3', 'CNMC_3_image_translation_f=0.4',
                'CNMC_3_image_translation_f=0.5', 'CNMC_3_image_translation_f=0.6', 'CNMC_3_image_translation_f=0.7',
                'CNMC_3_image_translation_f=0.8', 'CNMC_3_image_translation_f=0.9', 'CNMC_3_image_translation_f=1.0',
                'CNMC_3_image_zoom_f=0.1', 'CNMC_3_image_zoom_f=0.2', 'CNMC_3_image_zoom_f=0.3',
                'CNMC_3_image_zoom_f=0.4', 'CNMC_3_image_zoom_f=0.5', 'CNMC_3_image_zoom_f=0.6',
                'CNMC_3_image_zoom_f=0.7', 'CNMC_3_image_zoom_f=0.8', 'CNMC_3_image_zoom_f=0.9',
                'CNMC_3_image_zoom_f=1.0', 'CNMC_3_add_noise_gaussian_f=0.1', 'CNMC_3_add_noise_gaussian_f=0.2',
                'CNMC_3_add_noise_gaussian_f=0.3', 'CNMC_3_add_noise_gaussian_f=0.4', 'CNMC_3_add_noise_gaussian_f=0.5',
                'CNMC_3_add_noise_gaussian_f=0.6', 'CNMC_3_add_noise_gaussian_f=0.7', 'CNMC_3_add_noise_gaussian_f=0.8',
                'CNMC_3_add_noise_gaussian_f=0.9', 'CNMC_3_add_noise_gaussian_f=1.0', 'CNMC_3_add_noise_poisson_f=0.1',
                'CNMC_3_add_noise_poisson_f=0.2', 'CNMC_3_add_noise_poisson_f=0.3', 'CNMC_3_add_noise_poisson_f=0.4',
                'CNMC_3_add_noise_poisson_f=0.5', 'CNMC_3_add_noise_poisson_f=0.6', 'CNMC_3_add_noise_poisson_f=0.7',
                'CNMC_3_add_noise_poisson_f=0.8', 'CNMC_3_add_noise_poisson_f=0.9', 'CNMC_3_add_noise_poisson_f=1.0',
                'CNMC_3_add_noise_salt_and_pepper_f=0.1', 'CNMC_3_add_noise_salt_and_pepper_f=0.2',
                'CNMC_3_add_noise_salt_and_pepper_f=0.3', 'CNMC_3_add_noise_salt_and_pepper_f=0.4',
                'CNMC_3_add_noise_salt_and_pepper_f=0.5', 'CNMC_3_add_noise_salt_and_pepper_f=0.6',
                'CNMC_3_add_noise_salt_and_pepper_f=0.7', 'CNMC_3_add_noise_salt_and_pepper_f=0.8',
                'CNMC_3_add_noise_salt_and_pepper_f=0.9', 'CNMC_3_add_noise_salt_and_pepper_f=1.0',
                'CNMC_3_add_noise_speckle_f=0.1', 'CNMC_3_add_noise_speckle_f=0.2', 'CNMC_3_add_noise_speckle_f=0.3',
                'CNMC_3_add_noise_speckle_f=0.4', 'CNMC_3_add_noise_speckle_f=0.5', 'CNMC_3_add_noise_speckle_f=0.6',
                'CNMC_3_add_noise_speckle_f=0.7', 'CNMC_3_add_noise_speckle_f=0.8', 'CNMC_3_add_noise_speckle_f=0.9',
                'CNMC_3_add_noise_speckle_f=1.0', 'CNMC_3_imbalance_classes_f=0.1', 'CNMC_3_imbalance_classes_f=0.2',
                'CNMC_3_imbalance_classes_f=0.3', 'CNMC_3_imbalance_classes_f=0.4', 'CNMC_3_imbalance_classes_f=0.5',
                'CNMC_3_imbalance_classes_f=0.6', 'CNMC_3_imbalance_classes_f=0.7', 'CNMC_3_imbalance_classes_f=0.8',
                'CNMC_3_imbalance_classes_f=0.9', 'CNMC_3_imbalance_classes_f=1.0', 'CNMC_3_grayscale_f=0.1',
                'CNMC_3_grayscale_f=0.2', 'CNMC_3_grayscale_f=0.3', 'CNMC_3_grayscale_f=0.4',
                'CNMC_3_grayscale_f=0.5', 'CNMC_3_grayscale_f=0.6', 'CNMC_3_grayscale_f=0.7',
                'CNMC_3_grayscale_f=0.8', 'CNMC_3_grayscale_f=0.9', 'CNMC_3_grayscale_f=1.0',
                'CNMC_3_hsv_f=0.1', 'CNMC_3_hsv_f=0.2', 'CNMC_3_hsv_f=0.3', 'CNMC_3_hsv_f=0.4',
                'CNMC_3_hsv_f=0.5', 'CNMC_3_hsv_f=0.6', 'CNMC_3_hsv_f=0.7',
                'CNMC_3_hsv_f=0.8', 'CNMC_3_hsv_f=0.9', 'CNMC_3_hsv_f=1.0',
                'CNMC_3_blur_f=0.1', 'CNMC_3_blur_f=0.2', 'CNMC_3_blur_f=0.3', 'CNMC_3_blur_f=0.4',
                'CNMC_3_blur_f=0.5', 'CNMC_3_blur_f=0.6', 'CNMC_3_blur_f=0.7',
                'CNMC_3_blur_f=0.8', 'CNMC_3_blur_f=0.9', 'CNMC_3_blur_f=1.0',
                'CNMC_3_small_random_f=0.1', 'CNMC_3_small_random_f=0.2', 'CNMC_3_small_random_f=0.3', 'CNMC_3_small_random_f=0.4',
                'CNMC_3_small_random_f=0.5', 'CNMC_3_small_random_f=0.6', 'CNMC_3_small_random_f=0.7',
                'CNMC_3_small_random_f=0.8', 'CNMC_3_small_random_f=0.9', 'CNMC_3_small_random_f=1.0',
                'CNMC_3_small_easy_f=0.1', 'CNMC_3_small_easy_f=0.2', 'CNMC_3_small_easy_f=0.3', 'CNMC_3_small_easy_f=0.4',
                'CNMC_3_small_easy_f=0.5', 'CNMC_3_small_easy_f=0.6', 'CNMC_3_small_easy_f=0.7',
                'CNMC_3_small_easy_f=0.8', 'CNMC_3_small_easy_f=0.9', 'CNMC_3_small_easy_f=1.0',
                'CNMC_3_small_hard_f=0.1', 'CNMC_3_small_hard_f=0.2', 'CNMC_3_small_hard_f=0.3', 'CNMC_3_small_hard_f=0.4',
                'CNMC_3_small_hard_f=0.5', 'CNMC_3_small_hard_f=0.6', 'CNMC_3_small_hard_f=0.7',
                'CNMC_3_small_hard_f=0.8', 'CNMC_3_small_hard_f=0.9', 'CNMC_3_small_hard_f=1.0',
                'CNMC_3_small_clusters_f=0.1', 'CNMC_3_small_clusters_f=0.2', 'CNMC_3_small_clusters_f=0.3', 'CNMC_3_small_clusters_f=0.4',
                'CNMC_3_small_clusters_f=0.5', 'CNMC_3_small_clusters_f=0.6', 'CNMC_3_small_clusters_f=0.7',
                'CNMC_3_small_clusters_f=0.8', 'CNMC_3_small_clusters_f=0.9', 'CNMC_3_small_clusters_f=1.0',
                'CNMC_4', 'CNMC_4_image_rot_f=0.1', 'CNMC_4_image_rot_f=0.2',
                'CNMC_4_image_rot_f=0.3', 'CNMC_4_image_rot_f=0.4', 'CNMC_4_image_rot_f=0.5',
                'CNMC_4_image_rot_f=0.6', 'CNMC_4_image_rot_f=0.7', 'CNMC_4_image_rot_f=0.8',
                'CNMC_4_image_rot_f=0.9', 'CNMC_4_image_rot_f=1.0', 'CNMC_4_image_translation_f=0.1',
                'CNMC_4_image_translation_f=0.2', 'CNMC_4_image_translation_f=0.3', 'CNMC_4_image_translation_f=0.4',
                'CNMC_4_image_translation_f=0.5', 'CNMC_4_image_translation_f=0.6', 'CNMC_4_image_translation_f=0.7',
                'CNMC_4_image_translation_f=0.8', 'CNMC_4_image_translation_f=0.9', 'CNMC_4_image_translation_f=1.0',
                'CNMC_4_image_zoom_f=0.1', 'CNMC_4_image_zoom_f=0.2', 'CNMC_4_image_zoom_f=0.3',
                'CNMC_4_image_zoom_f=0.4', 'CNMC_4_image_zoom_f=0.5', 'CNMC_4_image_zoom_f=0.6',
                'CNMC_4_image_zoom_f=0.7', 'CNMC_4_image_zoom_f=0.8', 'CNMC_4_image_zoom_f=0.9',
                'CNMC_4_image_zoom_f=1.0', 'CNMC_4_add_noise_gaussian_f=0.1', 'CNMC_4_add_noise_gaussian_f=0.2',
                'CNMC_4_add_noise_gaussian_f=0.3', 'CNMC_4_add_noise_gaussian_f=0.4', 'CNMC_4_add_noise_gaussian_f=0.5',
                'CNMC_4_add_noise_gaussian_f=0.6', 'CNMC_4_add_noise_gaussian_f=0.7', 'CNMC_4_add_noise_gaussian_f=0.8',
                'CNMC_4_add_noise_gaussian_f=0.9', 'CNMC_4_add_noise_gaussian_f=1.0', 'CNMC_4_add_noise_poisson_f=0.1',
                'CNMC_4_add_noise_poisson_f=0.2', 'CNMC_4_add_noise_poisson_f=0.3', 'CNMC_4_add_noise_poisson_f=0.4',
                'CNMC_4_add_noise_poisson_f=0.5', 'CNMC_4_add_noise_poisson_f=0.6', 'CNMC_4_add_noise_poisson_f=0.7',
                'CNMC_4_add_noise_poisson_f=0.8', 'CNMC_4_add_noise_poisson_f=0.9', 'CNMC_4_add_noise_poisson_f=1.0',
                'CNMC_4_add_noise_salt_and_pepper_f=0.1', 'CNMC_4_add_noise_salt_and_pepper_f=0.2',
                'CNMC_4_add_noise_salt_and_pepper_f=0.3', 'CNMC_4_add_noise_salt_and_pepper_f=0.4',
                'CNMC_4_add_noise_salt_and_pepper_f=0.5', 'CNMC_4_add_noise_salt_and_pepper_f=0.6',
                'CNMC_4_add_noise_salt_and_pepper_f=0.7', 'CNMC_4_add_noise_salt_and_pepper_f=0.8',
                'CNMC_4_add_noise_salt_and_pepper_f=0.9', 'CNMC_4_add_noise_salt_and_pepper_f=1.0',
                'CNMC_4_add_noise_speckle_f=0.1', 'CNMC_4_add_noise_speckle_f=0.2', 'CNMC_4_add_noise_speckle_f=0.3',
                'CNMC_4_add_noise_speckle_f=0.4', 'CNMC_4_add_noise_speckle_f=0.5', 'CNMC_4_add_noise_speckle_f=0.6',
                'CNMC_4_add_noise_speckle_f=0.7', 'CNMC_4_add_noise_speckle_f=0.8', 'CNMC_4_add_noise_speckle_f=0.9',
                'CNMC_4_add_noise_speckle_f=1.0', 'CNMC_4_imbalance_classes_f=0.1', 'CNMC_4_imbalance_classes_f=0.2',
                'CNMC_4_imbalance_classes_f=0.3', 'CNMC_4_imbalance_classes_f=0.4', 'CNMC_4_imbalance_classes_f=0.5',
                'CNMC_4_imbalance_classes_f=0.6', 'CNMC_4_imbalance_classes_f=0.7', 'CNMC_4_imbalance_classes_f=0.8',
                'CNMC_4_imbalance_classes_f=0.9', 'CNMC_4_imbalance_classes_f=1.0', 'CNMC_4_grayscale_f=0.1',
                'CNMC_4_grayscale_f=0.2', 'CNMC_4_grayscale_f=0.3', 'CNMC_4_grayscale_f=0.4',
                'CNMC_4_grayscale_f=0.5', 'CNMC_4_grayscale_f=0.6', 'CNMC_4_grayscale_f=0.7',
                'CNMC_4_grayscale_f=0.8', 'CNMC_4_grayscale_f=0.9', 'CNMC_4_grayscale_f=1.0',
                'CNMC_4_hsv_f=0.1', 'CNMC_4_hsv_f=0.2', 'CNMC_4_hsv_f=0.3', 'CNMC_4_hsv_f=0.4',
                'CNMC_4_hsv_f=0.5', 'CNMC_4_hsv_f=0.6', 'CNMC_4_hsv_f=0.7',
                'CNMC_4_hsv_f=0.8', 'CNMC_4_hsv_f=0.9', 'CNMC_4_hsv_f=1.0',
                'CNMC_4_blur_f=0.1', 'CNMC_4_blur_f=0.2', 'CNMC_4_blur_f=0.3', 'CNMC_4_blur_f=0.4',
                'CNMC_4_blur_f=0.5', 'CNMC_4_blur_f=0.6', 'CNMC_4_blur_f=0.7',
                'CNMC_4_blur_f=0.8', 'CNMC_4_blur_f=0.9', 'CNMC_4_blur_f=1.0',
                'CNMC_4_small_random_f=0.1', 'CNMC_4_small_random_f=0.2', 'CNMC_4_small_random_f=0.3', 'CNMC_4_small_random_f=0.4',
                'CNMC_4_small_random_f=0.5', 'CNMC_4_small_random_f=0.6', 'CNMC_4_small_random_f=0.7',
                'CNMC_4_small_random_f=0.8', 'CNMC_4_small_random_f=0.9', 'CNMC_4_small_random_f=1.0',
                'CNMC_4_small_easy_f=0.1', 'CNMC_4_small_easy_f=0.2', 'CNMC_4_small_easy_f=0.3', 'CNMC_4_small_easy_f=0.4',
                'CNMC_4_small_easy_f=0.5', 'CNMC_4_small_easy_f=0.6', 'CNMC_4_small_easy_f=0.7',
                'CNMC_4_small_easy_f=0.8', 'CNMC_4_small_easy_f=0.9', 'CNMC_4_small_easy_f=1.0',
                'CNMC_4_small_hard_f=0.1', 'CNMC_4_small_hard_f=0.2', 'CNMC_4_small_hard_f=0.3', 'CNMC_4_small_hard_f=0.4',
                'CNMC_4_small_hard_f=0.5', 'CNMC_4_small_hard_f=0.6', 'CNMC_4_small_hard_f=0.7',
                'CNMC_4_small_hard_f=0.8', 'CNMC_4_small_hard_f=0.9', 'CNMC_4_small_hard_f=1.0',
                'CNMC_4_small_clusters_f=0.1', 'CNMC_4_small_clusters_f=0.2', 'CNMC_4_small_clusters_f=0.3', 'CNMC_4_small_clusters_f=0.4',
                'CNMC_4_small_clusters_f=0.5', 'CNMC_4_small_clusters_f=0.6', 'CNMC_4_small_clusters_f=0.7',
                'CNMC_4_small_clusters_f=0.8', 'CNMC_4_small_clusters_f=0.9', 'CNMC_4_small_clusters_f=1.0',
                'CNMC_5', 'CNMC_5_image_rot_f=0.1', 'CNMC_5_image_rot_f=0.2',
                'CNMC_5_image_rot_f=0.3', 'CNMC_5_image_rot_f=0.4', 'CNMC_5_image_rot_f=0.5',
                'CNMC_5_image_rot_f=0.6', 'CNMC_5_image_rot_f=0.7', 'CNMC_5_image_rot_f=0.8',
                'CNMC_5_image_rot_f=0.9', 'CNMC_5_image_rot_f=1.0', 'CNMC_5_image_translation_f=0.1',
                'CNMC_5_image_translation_f=0.2', 'CNMC_5_image_translation_f=0.3', 'CNMC_5_image_translation_f=0.4',
                'CNMC_5_image_translation_f=0.5', 'CNMC_5_image_translation_f=0.6', 'CNMC_5_image_translation_f=0.7',
                'CNMC_5_image_translation_f=0.8', 'CNMC_5_image_translation_f=0.9', 'CNMC_5_image_translation_f=1.0',
                'CNMC_5_image_zoom_f=0.1', 'CNMC_5_image_zoom_f=0.2', 'CNMC_5_image_zoom_f=0.3',
                'CNMC_5_image_zoom_f=0.4', 'CNMC_5_image_zoom_f=0.5', 'CNMC_5_image_zoom_f=0.6',
                'CNMC_5_image_zoom_f=0.7', 'CNMC_5_image_zoom_f=0.8', 'CNMC_5_image_zoom_f=0.9',
                'CNMC_5_image_zoom_f=1.0', 'CNMC_5_add_noise_gaussian_f=0.1', 'CNMC_5_add_noise_gaussian_f=0.2',
                'CNMC_5_add_noise_gaussian_f=0.3', 'CNMC_5_add_noise_gaussian_f=0.4', 'CNMC_5_add_noise_gaussian_f=0.5',
                'CNMC_5_add_noise_gaussian_f=0.6', 'CNMC_5_add_noise_gaussian_f=0.7', 'CNMC_5_add_noise_gaussian_f=0.8',
                'CNMC_5_add_noise_gaussian_f=0.9', 'CNMC_5_add_noise_gaussian_f=1.0', 'CNMC_5_add_noise_poisson_f=0.1',
                'CNMC_5_add_noise_poisson_f=0.2', 'CNMC_5_add_noise_poisson_f=0.3', 'CNMC_5_add_noise_poisson_f=0.4',
                'CNMC_5_add_noise_poisson_f=0.5', 'CNMC_5_add_noise_poisson_f=0.6', 'CNMC_5_add_noise_poisson_f=0.7',
                'CNMC_5_add_noise_poisson_f=0.8', 'CNMC_5_add_noise_poisson_f=0.9', 'CNMC_5_add_noise_poisson_f=1.0',
                'CNMC_5_add_noise_salt_and_pepper_f=0.1', 'CNMC_5_add_noise_salt_and_pepper_f=0.2',
                'CNMC_5_add_noise_salt_and_pepper_f=0.3', 'CNMC_5_add_noise_salt_and_pepper_f=0.4',
                'CNMC_5_add_noise_salt_and_pepper_f=0.5', 'CNMC_5_add_noise_salt_and_pepper_f=0.6',
                'CNMC_5_add_noise_salt_and_pepper_f=0.7', 'CNMC_5_add_noise_salt_and_pepper_f=0.8',
                'CNMC_5_add_noise_salt_and_pepper_f=0.9', 'CNMC_5_add_noise_salt_and_pepper_f=1.0',
                'CNMC_5_add_noise_speckle_f=0.1', 'CNMC_5_add_noise_speckle_f=0.2', 'CNMC_5_add_noise_speckle_f=0.3',
                'CNMC_5_add_noise_speckle_f=0.4', 'CNMC_5_add_noise_speckle_f=0.5', 'CNMC_5_add_noise_speckle_f=0.6',
                'CNMC_5_add_noise_speckle_f=0.7', 'CNMC_5_add_noise_speckle_f=0.8', 'CNMC_5_add_noise_speckle_f=0.9',
                'CNMC_5_add_noise_speckle_f=1.0', 'CNMC_5_imbalance_classes_f=0.1', 'CNMC_5_imbalance_classes_f=0.2',
                'CNMC_5_imbalance_classes_f=0.3', 'CNMC_5_imbalance_classes_f=0.4', 'CNMC_5_imbalance_classes_f=0.5',
                'CNMC_5_imbalance_classes_f=0.6', 'CNMC_5_imbalance_classes_f=0.7', 'CNMC_5_imbalance_classes_f=0.8',
                'CNMC_5_imbalance_classes_f=0.9', 'CNMC_5_imbalance_classes_f=1.0', 'CNMC_5_grayscale_f=0.1',
                'CNMC_5_grayscale_f=0.2', 'CNMC_5_grayscale_f=0.3', 'CNMC_5_grayscale_f=0.4',
                'CNMC_5_grayscale_f=0.5', 'CNMC_5_grayscale_f=0.6', 'CNMC_5_grayscale_f=0.7',
                'CNMC_5_grayscale_f=0.8', 'CNMC_5_grayscale_f=0.9', 'CNMC_5_grayscale_f=1.0',
                'CNMC_5_hsv_f=0.1', 'CNMC_5_hsv_f=0.2', 'CNMC_5_hsv_f=0.3', 'CNMC_5_hsv_f=0.4',
                'CNMC_5_hsv_f=0.5', 'CNMC_5_hsv_f=0.6', 'CNMC_5_hsv_f=0.7',
                'CNMC_5_hsv_f=0.8', 'CNMC_5_hsv_f=0.9', 'CNMC_5_hsv_f=1.0',
                'CNMC_5_blur_f=0.1', 'CNMC_5_blur_f=0.2', 'CNMC_5_blur_f=0.3', 'CNMC_5_blur_f=0.4',
                'CNMC_5_blur_f=0.5', 'CNMC_5_blur_f=0.6', 'CNMC_5_blur_f=0.7',
                'CNMC_5_blur_f=0.8', 'CNMC_5_blur_f=0.9', 'CNMC_5_blur_f=1.0',
                'CNMC_5_small_random_f=0.1', 'CNMC_5_small_random_f=0.2', 'CNMC_5_small_random_f=0.3', 'CNMC_5_small_random_f=0.4',
                'CNMC_5_small_random_f=0.5', 'CNMC_5_small_random_f=0.6', 'CNMC_5_small_random_f=0.7',
                'CNMC_5_small_random_f=0.8', 'CNMC_5_small_random_f=0.9', 'CNMC_5_small_random_f=1.0',
                'CNMC_5_small_easy_f=0.1', 'CNMC_5_small_easy_f=0.2', 'CNMC_5_small_easy_f=0.3', 'CNMC_5_small_easy_f=0.4',
                'CNMC_5_small_easy_f=0.5', 'CNMC_5_small_easy_f=0.6', 'CNMC_5_small_easy_f=0.7',
                'CNMC_5_small_easy_f=0.8', 'CNMC_5_small_easy_f=0.9', 'CNMC_5_small_easy_f=1.0',
                'CNMC_5_small_hard_f=0.1', 'CNMC_5_small_hard_f=0.2', 'CNMC_5_small_hard_f=0.3', 'CNMC_5_small_hard_f=0.4',
                'CNMC_5_small_hard_f=0.5', 'CNMC_5_small_hard_f=0.6', 'CNMC_5_small_hard_f=0.7',
                'CNMC_5_small_hard_f=0.8', 'CNMC_5_small_hard_f=0.9', 'CNMC_5_small_hard_f=1.0',
                'CNMC_5_small_clusters_f=0.1', 'CNMC_5_small_clusters_f=0.2', 'CNMC_5_small_clusters_f=0.3', 'CNMC_5_small_clusters_f=0.4',
                'CNMC_5_small_clusters_f=0.5', 'CNMC_5_small_clusters_f=0.6', 'CNMC_5_small_clusters_f=0.7',
                'CNMC_5_small_clusters_f=0.8', 'CNMC_5_small_clusters_f=0.9', 'CNMC_5_small_clusters_f=1.0',
                'CNMC_6', 'CNMC_6_image_rot_f=0.1', 'CNMC_6_image_rot_f=0.2',
                'CNMC_6_image_rot_f=0.3', 'CNMC_6_image_rot_f=0.4', 'CNMC_6_image_rot_f=0.5',
                'CNMC_6_image_rot_f=0.6', 'CNMC_6_image_rot_f=0.7', 'CNMC_6_image_rot_f=0.8',
                'CNMC_6_image_rot_f=0.9', 'CNMC_6_image_rot_f=1.0', 'CNMC_6_image_translation_f=0.1',
                'CNMC_6_image_translation_f=0.2', 'CNMC_6_image_translation_f=0.3', 'CNMC_6_image_translation_f=0.4',
                'CNMC_6_image_translation_f=0.5', 'CNMC_6_image_translation_f=0.6', 'CNMC_6_image_translation_f=0.7',
                'CNMC_6_image_translation_f=0.8', 'CNMC_6_image_translation_f=0.9', 'CNMC_6_image_translation_f=1.0',
                'CNMC_6_image_zoom_f=0.1', 'CNMC_6_image_zoom_f=0.2', 'CNMC_6_image_zoom_f=0.3',
                'CNMC_6_image_zoom_f=0.4', 'CNMC_6_image_zoom_f=0.5', 'CNMC_6_image_zoom_f=0.6',
                'CNMC_6_image_zoom_f=0.7', 'CNMC_6_image_zoom_f=0.8', 'CNMC_6_image_zoom_f=0.9',
                'CNMC_6_image_zoom_f=1.0', 'CNMC_6_add_noise_gaussian_f=0.1', 'CNMC_6_add_noise_gaussian_f=0.2',
                'CNMC_6_add_noise_gaussian_f=0.3', 'CNMC_6_add_noise_gaussian_f=0.4', 'CNMC_6_add_noise_gaussian_f=0.5',
                'CNMC_6_add_noise_gaussian_f=0.6', 'CNMC_6_add_noise_gaussian_f=0.7', 'CNMC_6_add_noise_gaussian_f=0.8',
                'CNMC_6_add_noise_gaussian_f=0.9', 'CNMC_6_add_noise_gaussian_f=1.0', 'CNMC_6_add_noise_poisson_f=0.1',
                'CNMC_6_add_noise_poisson_f=0.2', 'CNMC_6_add_noise_poisson_f=0.3', 'CNMC_6_add_noise_poisson_f=0.4',
                'CNMC_6_add_noise_poisson_f=0.5', 'CNMC_6_add_noise_poisson_f=0.6', 'CNMC_6_add_noise_poisson_f=0.7',
                'CNMC_6_add_noise_poisson_f=0.8', 'CNMC_6_add_noise_poisson_f=0.9', 'CNMC_6_add_noise_poisson_f=1.0',
                'CNMC_6_add_noise_salt_and_pepper_f=0.1', 'CNMC_6_add_noise_salt_and_pepper_f=0.2',
                'CNMC_6_add_noise_salt_and_pepper_f=0.3', 'CNMC_6_add_noise_salt_and_pepper_f=0.4',
                'CNMC_6_add_noise_salt_and_pepper_f=0.5', 'CNMC_6_add_noise_salt_and_pepper_f=0.6',
                'CNMC_6_add_noise_salt_and_pepper_f=0.7', 'CNMC_6_add_noise_salt_and_pepper_f=0.8',
                'CNMC_6_add_noise_salt_and_pepper_f=0.9', 'CNMC_6_add_noise_salt_and_pepper_f=1.0',
                'CNMC_6_add_noise_speckle_f=0.1', 'CNMC_6_add_noise_speckle_f=0.2', 'CNMC_6_add_noise_speckle_f=0.3',
                'CNMC_6_add_noise_speckle_f=0.4', 'CNMC_6_add_noise_speckle_f=0.5', 'CNMC_6_add_noise_speckle_f=0.6',
                'CNMC_6_add_noise_speckle_f=0.7', 'CNMC_6_add_noise_speckle_f=0.8', 'CNMC_6_add_noise_speckle_f=0.9',
                'CNMC_6_add_noise_speckle_f=1.0', 'CNMC_6_imbalance_classes_f=0.1', 'CNMC_6_imbalance_classes_f=0.2',
                'CNMC_6_imbalance_classes_f=0.3', 'CNMC_6_imbalance_classes_f=0.4', 'CNMC_6_imbalance_classes_f=0.5',
                'CNMC_6_imbalance_classes_f=0.6', 'CNMC_6_imbalance_classes_f=0.7', 'CNMC_6_imbalance_classes_f=0.8',
                'CNMC_6_imbalance_classes_f=0.9', 'CNMC_6_imbalance_classes_f=1.0', 'CNMC_6_grayscale_f=0.1',
                'CNMC_6_grayscale_f=0.2', 'CNMC_6_grayscale_f=0.3', 'CNMC_6_grayscale_f=0.4',
                'CNMC_6_grayscale_f=0.5', 'CNMC_6_grayscale_f=0.6', 'CNMC_6_grayscale_f=0.7',
                'CNMC_6_grayscale_f=0.8', 'CNMC_6_grayscale_f=0.9', 'CNMC_6_grayscale_f=1.0',
                'CNMC_6_hsv_f=0.1', 'CNMC_6_hsv_f=0.2', 'CNMC_6_hsv_f=0.3', 'CNMC_6_hsv_f=0.4',
                'CNMC_6_hsv_f=0.5', 'CNMC_6_hsv_f=0.6', 'CNMC_6_hsv_f=0.7',
                'CNMC_6_hsv_f=0.8', 'CNMC_6_hsv_f=0.9', 'CNMC_6_hsv_f=1.0',
                'CNMC_6_blur_f=0.1', 'CNMC_6_blur_f=0.2', 'CNMC_6_blur_f=0.3', 'CNMC_6_blur_f=0.4',
                'CNMC_6_blur_f=0.5', 'CNMC_6_blur_f=0.6', 'CNMC_6_blur_f=0.7',
                'CNMC_6_blur_f=0.8', 'CNMC_6_blur_f=0.9', 'CNMC_6_blur_f=1.0',
                'CNMC_6_small_random_f=0.1', 'CNMC_6_small_random_f=0.2', 'CNMC_6_small_random_f=0.3', 'CNMC_6_small_random_f=0.4',
                'CNMC_6_small_random_f=0.5', 'CNMC_6_small_random_f=0.6', 'CNMC_6_small_random_f=0.7',
                'CNMC_6_small_random_f=0.8', 'CNMC_6_small_random_f=0.9', 'CNMC_6_small_random_f=1.0',
                'CNMC_6_small_easy_f=0.1', 'CNMC_6_small_easy_f=0.2', 'CNMC_6_small_easy_f=0.3', 'CNMC_6_small_easy_f=0.4',
                'CNMC_6_small_easy_f=0.5', 'CNMC_6_small_easy_f=0.6', 'CNMC_6_small_easy_f=0.7',
                'CNMC_6_small_easy_f=0.8', 'CNMC_6_small_easy_f=0.9', 'CNMC_6_small_easy_f=1.0',
                'CNMC_6_small_hard_f=0.1', 'CNMC_6_small_hard_f=0.2', 'CNMC_6_small_hard_f=0.3', 'CNMC_6_small_hard_f=0.4',
                'CNMC_6_small_hard_f=0.5', 'CNMC_6_small_hard_f=0.6', 'CNMC_6_small_hard_f=0.7',
                'CNMC_6_small_hard_f=0.8', 'CNMC_6_small_hard_f=0.9', 'CNMC_6_small_hard_f=1.0',
                'CNMC_6_small_clusters_f=0.1', 'CNMC_6_small_clusters_f=0.2', 'CNMC_6_small_clusters_f=0.3', 'CNMC_6_small_clusters_f=0.4',
                'CNMC_6_small_clusters_f=0.5', 'CNMC_6_small_clusters_f=0.6', 'CNMC_6_small_clusters_f=0.7',
                'CNMC_6_small_clusters_f=0.8', 'CNMC_6_small_clusters_f=0.9', 'CNMC_6_small_clusters_f=1.0'],
        required=True,
        help='dataset, name should be like [dataset]_[split]_[modification]_f=[fraction], e.g. "ISIC_2_grayscale_f=0.4"',
        metavar='target dataset, name should be like [dataset]_[split]_[modification]_f=[fraction], e.g. "ISIC_2_grayscale_f=0.4"')
    parser.add_argument('-n',
        '--num_images',
        type=int,
        default=0,
        help='number of images to use for t-SNE plotting, if no number is given, all images are used')
    parser.add_argument('-m',
        '--mode',
        choices=['images', 'points'],
        default='images',
        help='what to plot, images or points. Default are images')
    parser.add_argument('-di',
        '--dims',
        choices=['2D', '3D'],
        default='2D',
        help='how many dimensions to plot. default is 2D')
    args = vars(parser.parse_args())

    main()
