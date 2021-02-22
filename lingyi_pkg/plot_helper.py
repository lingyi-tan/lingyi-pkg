import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

def plot_roc(labels, predict_prob):
    '''
    plot roc curve
    :param labels: y-true column
    :param predict_prob: y-predicted-prob column
    :return:
    '''
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, predict_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()
    return


def plot_pr(labels, predict_prob):
    '''
    plot precision-recall curve
    :param labels: y-true column
    :param predict_prob: y-predicted-prob column
    :return:
    '''
    precision, recall, thresholds = precision_recall_curve(labels, predict_prob)
    precision, recall = (list(t) for t in zip(*sorted(zip(precision, recall))))
    # add dummy end point
    precision.insert(0, 0.0)
    precision.append(1.0)
    recall.insert(0, 1.0)
    recall.append(0.0)
    pr_auc = auc(precision, recall)

    plt.title('PR')
    plt.plot(recall, precision, 'r', label='PR-AUC = %0.4f' % pr_auc)
    plt.legend(loc='upper right')
    plt.ylabel('P')
    plt.xlabel('R')
    plt.show()

    return