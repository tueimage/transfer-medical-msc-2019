from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
import config_ISIC
import os
import pickle

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
