'''
purpose: genereate figures for the paper draft
overlap heatmaps on wsi for visulization

author: Hongming Xu
email: mxu@ualberta.ca
'''

import sys
import os
import matplotlib.pyplot as plt
import skimage.transform
rela_path='../../../'
sys.path.insert(0,rela_path+'xhm_deep_learning/functions')
from read_wsi_mag import read_wsi_mag
from overlap_heatmap import overlap_heatmap

def overlap_wsi(wsi_file,pred_file):

    LR=read_wsi_mag(wsi_file,1.25)

    pred=plt.imread(pred_file)

    pred=pred[:,:,0]

    pred_mask=skimage.transform.resize(pred,LR.shape[0:2],order=0) # 0 nearest-neighbor, 1: bi-linear (default)

    overlap_heatmap(LR, pred_mask, './' + 'example.png')


if __name__=='__main__':
    wsi_path = rela_path+'data/tcga_coad_slide/tcga_coad/quality_uncertain/'

    pred_path1 = rela_path+'data/tcga_coad_read_data/coad_tumor_preds/resnet18_tcga_v2_tils/'

    pred_path2 =  rela_path+'data/tcga_coad_read_data/coad_read_tils_preds/pred_maps_0.4/'

    wsis = sorted(os.listdir(wsi_path))
    for img_name in wsis[12:]:
        if '.svs' in img_name:
            file = wsi_path + img_name
            #pred = pred_path1 + img_name[0:23] + '.png'
            pred = pred_path2+ img_name[0:23] + '_gray.png'
            overlap_wsi(file, pred)