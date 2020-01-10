'''
overlap tils detection on color wsi
observe if thresholded tils regions are correct
purpose: used to check tils threshold
only used duing developing studies

author: Hongming Xu, CCF
email: mxu@ualberta.ca
'''
import os
import time
import scipy.misc
import skimage.transform
import skimage.segmentation
from skimage.io import imsave
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.insert(0,'../../../xhm_deep_learning/functions')
from read_wsi_mag import read_wsi_mag

def save_wsi_tils(File,pred_file):
    since = time.time()
    LR=read_wsi_mag(File,1.25/2)

    pred=plt.imread(pred_file)
    pred_mask=pred[:,:,0]
    pred_mask2=skimage.transform.resize(pred_mask,LR.shape[0:2],order=0) # 0 nearest-neighbor, 1: bi-linear (default)
    pred_mask3=skimage.segmentation.find_boundaries(pred_mask2)

    LR2=LR.copy()
    LR2[pred_mask3!=0]=[0,0,255]
    imsave(pred_file.split('/')[-1] + '.png',LR2)

    # f = plt.figure()
    # plt.imshow(LR)
    # plt.contour(pred_mask2, [0.5], colors=['blue'],linewidths=[0.4])
    # plt.savefig(pred_file.split('/')[-1] + '.png')
    # f.clear()
    # plt.close()

    time_elapsed = time.time() - since
    print('Complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


if __name__=='__main__':
    imagePath=['../../../data/kang_colon_slide/181119/']
    destPath=['../../../data/pan_cancer_tils/data_yonsei_v01_pred/181119/']
    for i in range(len(imagePath)):
        temp_imagePath = imagePath[i]
        dest_imagePath = destPath[i]
        wsis = sorted(os.listdir(temp_imagePath))
        for img_name in wsis[:15]:
            if '.mrxs' in img_name:
                file=temp_imagePath+img_name
                pred=dest_imagePath+img_name.split('.')[0]+'_color.png'
                save_wsi_tils(file,pred)
