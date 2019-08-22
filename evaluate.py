import numpy as np
import os
import pandas
import matplotlib.pyplot as plt
import json
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score

def ROC_AUC(preds, true_labels, config, timestamp):
    # initialize TPR, FPR, ACC and AUC lists
    TPR_list, FPR_list, ACC_list = [], [], []
    AUC_score = []

    # preds is an array like [[x] [x] [x]], make it into array like [x x x]
    preds = np.asarray([label for sublist in preds for label in sublist])

    # calculate for different thresholds
    thresholds = -np.sort(-(np.unique(preds)))
    for threshold in thresholds:
        # apply threshold to predictions
        pred_labels = np.where(preds > threshold, 1, 0).astype(int)

        # calculate True Positive (TP), True Negative (TN), False Positive (FP) and
        # False Negative (FN)
        TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
        TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
        FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
        FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))

        # calculate TPR, FPR, ACC and add to lists
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        ACC = (TP + TN) / (TP + TN + FP + FN)

        TPR_list.append(TPR)
        FPR_list.append(FPR)
        ACC_list.append(ACC)

        AUC_score.append((1-FPR+TPR)/2)

        pred_labels = []

    AUC = round(sum(AUC_score)/len(thresholds),3)
    print("AUC: {}".format(AUC))

    AUC2 = round(roc_auc_score(true_labels, preds),3)
    print("sk AUC: {}".format(AUC2))

    # plot and save ROC curve
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(FPR_list, TPR_list)
    plt.plot([0,1],[0,1], '--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.title("ROC Curve, AUC = {}".format(AUC))
    plt.savefig(os.path.join(config['plot_path'], "{}_ROC.png".format(timestamp)))

    # also plot accuracies for each threshold
    plt.figure()
    plt.plot(thresholds, ACC_list)
    plt.plot([0,1],[0,1], '--')
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.title("Accuracy per threshold")
    plt.savefig(os.path.join(config['plot_path'], "{}_ACC.png".format(timestamp)))

    # save plot data in csv file
    csvpath = os.path.join(config['model_savepath'], '{}_eval.csv'.format(timestamp))
    pandas.DataFrame([TPR_list, FPR_list, ACC_list]).to_csv(csvpath)
