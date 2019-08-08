import numpy as np
import os
import matplotlib.pyplot as plt

def ROC_AUC(preds, true_labels, plotpath, timestamp):
    # initialize TPR, FPR, ACC and AUC lists
    TPR_list, FPR_list, ACC_list = [], [], []
    AUC_score = []
    # calculate for different thresholds
    thresholds = np.linspace(1.0, 0.0, num=21, endpoint=True)
    for threshold in thresholds:
        pred_labels = preds
        pred_labels = np.where(pred_labels > threshold, 1, 0)

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

    print(TPR_list)
    print(FPR_list)
    print(ACC_list)

    AUC = sum(AUC_score)/len(thresholds)

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
    plt.savefig(os.path.join(plotpath, "{}_ROC.png".format(timestamp)))

    # also plot accuracies for each threshold
    plt.figure()
    plt.plot(thresholds, ACC_list)
    plt.plot([0,1],[0,1], '--')
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.title("Accuracy per threshold")
    plt.savefig(os.path.join(plotpath, "{}_ACC.png".format(timestamp)))
