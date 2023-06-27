import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import numpy as np
import scikitplot as skplt

def draw_roc_curve(y, y_pred):
    fpr, tpr, thresholds= roc_curve(y, y_pred)                         #tpr, fpr, threshold roc_curve 사용하여 구하기

    # ix = np.argmin(fpr-tpr)                                             #가장 차이가 작은 값에서 인덱스 값, optimal threshold
    ix = np.argmax(tpr-fpr)
    # ix = np.argmin(np.linalg.norm([0,1],[fpr,tpr]))                      #왼쪽 1.0이랑 가장 가까운 값을 찾으면서 보기
    best_thresh = thresholds[ix]

    sens, spec = tpr[ix], 1-fpr[ix]
    auc = roc_auc_score(y, y_pred)

    plt.plot([0, 1], [0, 1], linestyle='--', markersize=0.01, color='black')               #글씨 사이즈, 색깔 바꾸기
    plt.plot(fpr, tpr, marker='.', color='blue', markersize=0.05)
    plt.scatter(fpr[ix], tpr[ix], marker='+', s=100, color='r',
                label='Best threshold = %.3f, \nSensitivity = %.3f, \nSpecificity = %.3f' % (best_thresh, sens, spec))

    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=4)

    # show the plot
    plt.show()

    y_pred[y_pred < best_thresh] = 0  # thresh 보다 작으면 0
    y_pred[y_pred >= best_thresh] = 1  # 크거나 같으면 1

    skplt.metrics.plot_confusion_matrix(y, y_pred, normalize=False, text_fontsize="large")
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    prec = tp / (tp + fp)
    sens = tp / (tp + fn)
    f1 = 2 * prec * sens / (prec + sens)
    f1 = np.round(f1 * 100, 4)
    acc = (tp + tn) / (tp + fn + tn + fp)
    acc = np.round(acc * 100, 4)
    spec = tn / (tn + fp)
    spec = np.round(spec * 100, 4)
    sens = np.round(sens * 100, 4)
    roc_auc = np.round(auc, 4)


    plt.xlabel('Predicted label', fontsize=15)
    plt.ylabel('True label', fontsize=15)
    plt.title('Normalized Confusion Matrix', fontsize=15)
    plt.show()
    print(f"Accuracy: {acc}% / Sensitivity: {sens}% / Specificity: {spec}% / F1-score: {f1}% / AUC: {roc_auc}")
    draw_plt(acc, sens, spec, roc_auc, f1)



def draw_plt(accuracy, recall, precision, AUC, f1_score):

    #점수 지표 그리기
    fig, ax = plt.subplots(1,1)
    data = [[round(accuracy, 4), round(recall, 4), round(precision, 4), round(AUC, 4), round(f1_score, 4)]]
    column_label = ("accuracy", "recall", "precision", "AUC", "f1_score")
    ax.axis('tight')
    ax.axis('off')

    ax.table(cellText = data, colLabels=column_label, loc="center")
    plt.show()

