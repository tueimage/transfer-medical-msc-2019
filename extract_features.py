from sklearn.preprocessing import LabelEncoder
from keras.applications import VGG16
from keras.applications import imagenet_utils
import cv2
import numpy as np
import config_ISIC
import pickle
import os
import glob
import random
random.seed(0)

# load VGG16 network with ImageNet weights
# without last fully connected layer with softmax
print("loading network...")
model = VGG16(weights="imagenet", include_top=False)
labelencoder = None

# loop over data splits
for split in (config_ISIC.TRAIN, config_ISIC.VAL, config_ISIC.TEST):
    # get all image paths in current split and shuffle them
    print("processing '{} split'...".format(split))
    path = os.path.join(config_ISIC.DATASET_PATH, split)
    imagepaths = glob.glob(os.path.join(path, '**/*.jpg'))
    random.shuffle(imagepaths)

    # get a list with corresponding labels and create labelencoder if it's None
    labels = [path.split(os.path.sep)[-2] for path in imagepaths]
    if labelencoder is None:
        labelencoder = LabelEncoder()
        labelencoder.fit(labels)

    # create output folder if it doesn't exist yet
    if not os.path.exists(config_ISIC.BASE_CSV_PATH):
        os.makedirs(config_ISIC.BASE_CSV_PATH)

    # create output csv file path and open it for writing
    csvpath = os.path.join(config_ISIC.BASE_CSV_PATH, "{}.csv".format(split))
    with open(csvpath, 'w') as csv:
        # loop over the images in batches
        for (b, i) in enumerate(range(0, len(imagepaths), config_ISIC.BATCH_SIZE)):
            # extract batch of images and labels
            print("processing batch {}/{}".format(b+1,
                int(np.ceil(len(imagepaths) / float(config_ISIC.BATCH_SIZE)))))
            batchpaths = imagepaths[i:i + config_ISIC.BATCH_SIZE]
            batchlabels = labelencoder.transform(labels[i:i + config_ISIC.BATCH_SIZE])

            # initialize list of images
            batchimages = []

            for imagepath in batchpaths:
                # load input image and resize to (224, 224)
                image = cv2.imread(imagepath)
                image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_NEAREST)

                # expand the dimensions, channels first and subtract mean RGB
                # pixel intensity from the ImageNet dataset
                image = np.expand_dims(image, axis=0)
                image = imagenet_utils.preprocess_input(image)

                # add image to batch
                batchimages.append(image)

            # pass batch of images through network, the output will be our features
            batchimages = np.vstack(batchimages)
            features = model.predict(batchimages, batch_size=config_ISIC.BATCH_SIZE)

            # output shape of max-pooling layer is (batch_size, 7, 7, 512)
            # reshape features to a flattened volume
            features = features.reshape((features.shape[0], 7 * 7 * 512))

            # loop over class labels and extracted features to add to csv
            for (label, feat) in zip(batchlabels, features):
                # transform feature list into a comma separated row
                # and write to csv
                feat = ",".join([str(f) for f in feat])
                csv.write("{},{}\n".format(label,feat))

# save label encoder
with open(config_ISIC.LE_PATH, 'wb') as le_file:
    le_file.write(pickle.dumps(labelencoder))
