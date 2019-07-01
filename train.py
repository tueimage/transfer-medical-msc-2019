from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers import Input
from keras.optimizers import SGD, RMSprop
import matplotlib.pyplot as plt
import numpy as np
import config_ISIC
import os
import pickle
import glob
import models

# choose GPU for training
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def load_data(splitpath):
    data, labels = [], []

    # loop over the rows in data split file with extracted features
    for row in open(splitpath):
        # extract class label and features and add to lists
        row = row.strip().split(",")
        label = row[0]
        features = np.array(row[1:], dtype="float")

        data.append(features)
        labels.append(label)

    # convert lists to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    return (data, labels)

def plot_training(hist, epochs, plotpath):
    # plot and save training history
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), hist.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), hist.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), hist.history["acc"], label="train_acc")
    plt.plot(np.arange(0, epochs), hist.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper right")
    plt.savefig(plotpath)

from_scratch = True
feature_extraction = False
fine_tuning = False

if from_scratch:
    # get paths to training, validation and testing directories
    trainingpath = os.path.join(config_ISIC.DATASET_PATH, config_ISIC.TRAIN)
    validationpath = os.path.join(config_ISIC.DATASET_PATH, config_ISIC.VAL)
    testpath = os.path.join(config_ISIC.DATASET_PATH, config_ISIC.TEST)

    # get total number of images in each split, needed to train in batches
    num_training = len(glob.glob(os.path.join(trainingpath, '**/*.jpg')))
    num_validation = len(glob.glob(os.path.join(validationpath, '**/*.jpg')))
    num_test = len(glob.glob(os.path.join(testpath, '**/*.jpg')))

    # initialize image data generator objects
    gen_obj_training = ImageDataGenerator(rescale=1./255)
    gen_obj_test = ImageDataGenerator(rescale=1./255)

    # initialize the image generators that load batches of images
    gen_training = gen_obj_training.flow_from_directory(
        trainingpath,
        class_mode="binary",
        target_size=(224,224),
        color_mode="rgb",
        shuffle=True,
        batch_size=config_ISIC.BATCH_SIZE)

    gen_validation = gen_obj_test.flow_from_directory(
        validationpath,
        class_mode="binary",
        target_size=(224,224),
        color_mode="rgb",
        shuffle=False,
        batch_size=config_ISIC.BATCH_SIZE)

    gen_test = gen_obj_test.flow_from_directory(
        testpath,
        class_mode="binary",
        target_size=(224,224),
        color_mode="rgb",
        shuffle=False,
        batch_size=config_ISIC.BATCH_SIZE)

    # set input tensor for VGG16 model
    input_tensor = Input(shape=(224,224,3))

    # load VGG16 model architecture
    model_VGG16 = models.model_VGG16(input_tensor)
    print(model_VGG16.summary())

    # set optimizer and compile model
    print("compiling model...")
    sgd = SGD(lr=0.01, momentum=0.9)
    RMSprop = RMSprop(lr=0.01)
    model_VGG16.compile(loss="binary_crossentropy", optimizer=RMSprop, metrics=["accuracy"])

    # calculate relative class weights for the imbalanced training data
    class_weights = {}
    for i in range(len(config_ISIC.CLASSES)):
        # get path to the class images and get number of samples for that class
        classpath = os.path.join(trainingpath, config_ISIC.CLASSES[i])
        num_class = len(glob.glob(os.path.join(classpath, '*.jpg')))

        # calculate relative class weight and add to dictionary
        class_weight = num_training/num_class - 1
        class_weights[i] = class_weight

    # train the model
    print("training model...")
    hist = model_VGG16.fit_generator(
        gen_training,
        steps_per_epoch = num_training // config_ISIC.BATCH_SIZE,
        validation_data = gen_validation,
        validation_steps = num_validation // config_ISIC.BATCH_SIZE,
        class_weight=class_weights,
        epochs=10,
        verbose=1)

    # create save directory if it doesn't exist and save trained model
    print("saving model...")
    if not os.path.exists(config_ISIC.MODEL_SAVEPATH):
        os.makedirs(config_ISIC.MODEL_SAVEPATH)
    savepath = os.path.join(config_ISIC.MODEL_SAVEPATH, "model_VGG16.h5")
    model_VGG16.save(savepath)

    # create plot directory if it doesn't exist and plot training progress
    print("saving plots...")
    if not os.path.exists(config_ISIC.PLOT_PATH):
        os.makedirs(config_ISIC.PLOT_PATH)
    plotpath = os.path.join(config_ISIC.PLOT_PATH, "training.png")
    plot_training(hist, 10, plotpath)

    # now we need to check the model on the validation data and use this for tweaking (not on test data)
    # this is for checking the best training settings; afterwards we can test on test set
    print("evaluating model...")

    # make predictions and take highest predicted value as class label
    preds = model_VGG16.predict_generator(gen_validation, steps=(num_validation//config_ISIC.BATCH_SIZE), verbose=1)
    preds = np.argmax(preds, axis=1)

    # make a classification report (ROC/AUC?)


if feature_extraction:
    # get paths to training and test csv files
    trainingpath = os.path.join(config_ISIC.BASE_CSV_PATH, "{}.csv".format(config_ISIC.TRAIN))
    testpath = os.path.join(config_ISIC.BASE_CSV_PATH, "{}.csv".format(config_ISIC.TEST))

    # load data from disk
    print("loading data...")
    (Xtrain, Ytrain) = load_data(trainingpath)
    (Xtest, Ytest) = load_data(testpath)

    # load label encoder
    labelencoder = pickle.loads(open(config_ISIC.LE_PATH, 'rb').read())

    # train the model
    print("training model...")
    model = LogisticRegression(solver="liblinear", multi_class="auto")
    model.fit(Xtrain, Ytrain)

    # evaluate model
    print("evaluating model...")
    preds = model.predict(Xtest)
    print(classification_report(Ytest, preds, target_names=labelencoder.classes_))

    # save model
    print("saving model...")
    with open(config_ISIC.MODEL_PATH, 'wb') as model_file:
        model_file.write(pickle.dumps(model))

if fine_tuning:
    # get paths to training, validation and testing directories
    trainingpath = os.path.join(config_ISIC.DATASET_PATH, config_ISIC.TRAIN)
    validationpath = os.path.join(config_ISIC.DATASET_PATH, config_ISIC.VAL)
    testpath = os.path.join(config_ISIC.DATASET_PATH, config_ISIC.TEST)

    # get total number of images in each split, needed to train in batches
    num_training = len(glob.glob(os.path.join(trainingpath, '**/*.jpg')))
    num_validation = len(glob.glob(os.path.join(validationpath, '**/*.jpg')))
    num_test = len(glob.glob(os.path.join(testpath, '**/*.jpg')))

    # initialize image data generator objects
    gen_obj_training = ImageDataGenerator()
    gen_obj_test = ImageDataGenerator()

    # add mean subtraction with ImageNet mean to the generator
    imagenet_mean = np.array([123.68, 116.779, 103.939], dtype="float32")
    gen_obj_training.mean = imagenet_mean
    gen_obj_test.mean = imagenet_mean

    # initialize the image generators that load batches of images
    gen_training = gen_obj_training.flow_from_directory(
        trainingpath,
        class_mode="categorical",
        target_size=(224,224),
        color_mode="rgb",
        shuffle=True,
        batch_size=config_ISIC.BATCH_SIZE)

    gen_validation = gen_obj_test.flow_from_directory(
        validationpath,
        class_mode="categorical",
        target_size=(224,224),
        color_mode="rgb",
        shuffle=False,
        batch_size=config_ISIC.BATCH_SIZE)

    gen_test = gen_obj_test.flow_from_directory(
        testpath,
        class_mode="categorical",
        target_size=(224,224),
        color_mode="rgb",
        shuffle=False,
        batch_size=config_ISIC.BATCH_SIZE)

    # now load the VGG16 network with ImageNet weights
    # without last fully connected layer with softmax
    print("loading network...")
    base_model = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224,224,3)))

    # build classifier model to put on top of the base model
    top_model = base_model.output
    top_model = Flatten()(top_model)
    top_model = Dense(32, activation="relu")(top_model)
    top_model = Dropout(0.5)(top_model)
    top_model = Dense(len(config_ISIC.CLASSES), activation="softmax")(top_model)

    # add the model on top of the base model
    model = Model(inputs=base_model.input, outputs=top_model)

    # freeze all layers in the base model to exclude them from training
    for layer in base_model.layers:
        layer.trainable = False

    # compile model
    print("compiling model...")
    sgd = SGD(lr=1e-4, momentum=0.9)
    model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])

    # train model (only the top) for a few epochs so the new layers get
    # initialized with learned values instead of randomly
    print("training top model...")
    hist = model.fit_generator(
        gen_training,
        steps_per_epoch = num_training // config_ISIC.BATCH_SIZE,
        validation_data = gen_validation,
        validation_steps = num_validation // config_ISIC.BATCH_SIZE,
        epochs=5,
        verbose=1)

    # reset the testing generator for network evaluation using the test data
    print("evaluating after fine-tuning top model...")
    gen_test.reset()

    # make predictions and take highest predicted value as class label
    preds = model.predict_generator(gen_test, steps=(num_test//config_ISIC.BATCH_SIZE), verbose=1)
    preds = np.argmax(preds, axis=1)

    # print classification report
    print(classification_report(gen_test.classes, preds, target_names=gen_test.class_indices.keys()))

    # create plot directory if it doesn't exist and plot training progress
    if not os.path.exists(config_ISIC.PLOT_PATH):
        os.makedirs(config_ISIC.PLOT_PATH)
    plotpath = os.path.join(config_ISIC.PLOT_PATH, "warmup_training.png")
    plot_training(hist, 5, plotpath)

    # now we can unfreeze base model layers to train more
    # unfreeze the last convolutional layer in VGG16
    for layer in base_model.layers[15:]:
        layer.trainable = True

    # print which layers are trainable now
    for layer in base_model.layers:
        print("{}: {}".format(layer, layer.trainable))

    # reset image generators before training again
    gen_training.reset()
    gen_validation.reset()

    # recompile the model
    print("recompiling model...")
    sgd = SGD(lr=1e-4, momentum=0.9)
    model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])

    # train the model again, with extra trainable layers
    print("training recompiled model...")
    hist = model.fit_generator(
        gen_training,
        steps_per_epoch = num_training // config_ISIC.BATCH_SIZE,
        validation_data = gen_validation,
        validation_steps = num_validation // config_ISIC.BATCH_SIZE,
        epochs=5,
        verbose=1)

    # and evaluate again
    print("evaluating after fine-tuning network...")
    gen_test.reset()
    preds = model.predict_generator(gen_test, steps=(num_test//config_ISIC.BATCH_SIZE), verbose=1)
    preds = np.argmax(preds, axis=1)
    print(classification_report(gen_test.classes, preds, target_names=gen_test.class_indices.keys()))
    plotpath = os.path.join(config_ISIC.PLOT_PATH, "unfrozen_training.png")
    plot_training(hist, 5, plotpath)
