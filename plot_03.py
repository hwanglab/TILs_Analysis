'''
purpose: plot ROC curve of tils detection on internal testing set

author: Hongming XU, CCF, 2020
email: mxu@ualberta.ca

'''
import pandas as pd
import glob
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

test_path=['../../../data/pan_cancer_tils/data_v02/test/tils/',
           '../../../data/pan_cancer_tils/data_v02/test/others/']
resnet18_output='../../../data/pan_cancer_tils/models_v02/'
shufflenet_output='../../../data/pan_cancer_tils/models_v03/'
resnet34_output='../../../data/pan_cancer_tils/models_v04/'


def roc_evaluation(y_true,y_score):
    fpr,tpr,thresholds=roc_curve(y_true,y_score,pos_label='tils')
    auc=roc_auc_score(y_true=='tils',y_score)
    return fpr,tpr,thresholds,auc

def extract_gt(pred_file):
    df = pd.read_excel(pred_file)
    img_names = df['Name']
    preds = df['Pred']
    gt = []
    for ind, temp_img in enumerate(img_names):
        tils = glob.glob(test_path[0] + temp_img)
        tils_bad = glob.glob(test_path[0] + 'bad_quality/' + temp_img)
        others = glob.glob(test_path[1] + temp_img)
        if len(tils) == 1 and len(others) == 0:
            gt.append('tils')
        elif len(tils_bad) == 1 and len(others) == 0:
            gt.append('tils')
        elif len(tils) == 0 and len(others) == 1:
            gt.append('others')
        else:
            raise RuntimeError(f'unexpected result{(ind, temp_img)}')

    return gt, preds

if __name__=='__main__':

    best_models=[resnet18_output+'resnet18_0_adam_0.0001_4.pt.xlsx',
                 shufflenet_output+'shufflenet_0_adam_0.001_4.pt.xlsx',
                 resnet34_output+'resnet34_0_adam_0.001_32.pt.xlsx']
    colors=['darkorange','darkviolet','darkgreen']
    labels=['resnet18','shufflenet','resnet34']
    plt.figure()
    lw = 2
    for ind,temp_file in enumerate(best_models):
        gt,preds=extract_gt(temp_file)
        fpr, tpr, thresholds, auc = roc_evaluation(np.asarray(gt), np.asarray(preds))
        optimal_idx = np.argmax(np.abs(tpr - fpr))
        optimal_threshold = thresholds[optimal_idx]


        plt.plot(fpr, tpr, color=colors[ind],
                 lw=lw, label='%s (auc = %0.2f)' % (labels[ind],auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic curve')
        plt.legend(loc="lower right")
        print("auc=%f" % (auc))
    plt.show()

