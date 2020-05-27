import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import cv2
import tensorflow as tf
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, roc_auc_score
from keras import backend as K
import tensorflow


def perform_PCA(features, pca_pct=.95):
    """Reduce dimensionality of dataset features."""
    # first scale the data to zero mean, unit variance
    scaler = StandardScaler()
    scaler.fit(features)
    features = scaler.transform(features)

    # fit PCA
    pca = PCA(pca_pct)
    pca.fit(features)
    pca.n_components_

    # apply PCA to features
    features = pca.transform(features)

    return features


def limit_memory():
    """Release unused memory sources, force garbage collection."""
    K.clear_session()
    K.get_session().close()
    tf.reset_default_graph()
    gc.collect()
    K.set_session(tf.Session())
    gc.collect()


def AUC_score(preds, true_labels):
    """Calculate AUC score."""
    fpr, tpr, thresholds = roc_curve(true_labels, preds, pos_label=1)

    AUC = round(roc_auc_score(true_labels, preds), 3)
    print("AUC: {}".format(AUC))

    return fpr, tpr, thresholds, AUC


def accuracy(preds, true_labels):
    """Calculate Accuracy."""
    pred_labels = np.where(preds > 0.5, 1, 0).astype(int)

    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))

    acc = round(((TP + TN) / (TP + TN + FP + FN)), 3)
    print("ACC: {}".format(acc))

    return acc


def plot_AUC(fpr, tpr, AUC, savepath):
    """Plot and save ROC curve."""
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("ROC Curve, AUC = {}".format(AUC))
    plt.savefig(savepath)


def load_training_data(trainingpath):
    """Load training data."""
    images = []
    imagepaths = glob.glob(os.path.join(trainingpath, '**/*.jpg'))

    for path in imagepaths:
        images.append(cv2.imread(path))

    images = np.array(images)
    return images
