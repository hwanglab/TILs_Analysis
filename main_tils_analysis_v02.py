'''
main function to analyze tumor-til-maps
author: Hongming Xu, 2020, CCF
email: mxu@ualberta.ca

purpose: analyze tumor-tils architectural features

input:
    wsi image
    tumor prediction map
    til prediction map
output:
    feature vector for the corresponding wsi
'''

import os
import numpy as np
import openslide
import scipy
import time
import matplotlib.pyplot as plt
from PIL import Image
import concurrent.futures
from itertools import repeat
import pandas as pd
import time
from tqdm import tqdm
from skimage import measure, transform, morphology
from scipy import ndimage
from skimage.draw import circle
import sys
sys.path.insert(0,'../../xhm_deep_learning/functions')
from wsi_preprocess_mask import wsi_preprocess_mask_v02
from wsi_coarse_level import wsi_coarse_read


def iterate_circles(binary_mask):
    '''
    input: binary mask, where object foreground is binary 1
    output:
        circle_mask: binary circle mask
        circle_mask2: circle mask overlapped on binary mask (for debuging observation)
    '''
    circle_mask2 = binary_mask.astype(float)
    circle_mask = np.zeros_like(binary_mask)
    max_rad = np.Inf
    while max_rad>2:
        dis_map=ndimage.distance_transform_edt(binary_mask)
        max_rad=np.max(dis_map)
        r_c=np.where(dis_map==np.max(dis_map))
        rr, cc = circle(float(r_c[0][0]), float(r_c[1][0]), np.floor(max_rad - 1))
        # if max_rad>2:
        #     rr,cc=circle(float(r_c[0][0]),float(r_c[1][0]),np.floor(max_rad-1))
        # else:
        #     rr,cc=r_c[0][0], r_c[1][0]

        bool_indicator=np.logical_or(np.logical_or(rr>circle_mask.shape[0]-1, rr<0), np.logical_or(cc>circle_mask.shape[1]-1,cc<0))
        ind=np.asarray(np.where(bool_indicator))
        if ind.size>0:
            rr=np.delete(rr,ind)
            cc=np.delete(cc,ind)



        circle_mask[rr, cc] = 1
        circle_mask2[rr, cc] = 2

        binary_mask=np.logical_and(binary_mask,np.invert(circle_mask))

    return circle_mask,circle_mask2

def tumor_til_analysis_v02(file_img,file_tumor,file_til,thr,mag):
    # open slide
    Slide=openslide.OpenSlide(file_img)
    #LR, Objective, pxy=wsi_coarse_read(Slide,mag) # at 2.5 magnification

    # open til mask
    til_map=plt.imread(file_til)
    til_mask=til_map[:,:,0]
    til_mask = morphology.remove_small_holes(til_mask.astype(bool), 4)
    # til_mask = transform.rescale(til_mask,8, order=0)

    #tissue_mask = np.logical_or(til_mask,til_map[:,:,2])
    #til_mask=transform.resize(til_mask,LR.shape[0:2],order=0) # order=0 nearest-neighbor
    #tissue_mask=transform.resize(tissue_mask,LR.shape[0:2],order=0)

    # open tumor mask
    tumor_mask = plt.imread(file_tumor)
    tumor_mask = transform.resize(tumor_mask, til_mask.shape[0:2], order=1)  # order=0 nearest-neighbor
    tumorb = wsi_preprocess_mask_v02(tumor_mask, thr)

    # analyze the largest tumor
    mask_label = measure.label(tumorb, neighbors=8, background=0)
    properties = measure.regionprops(mask_label)
    if len(properties) > 1:
        thrNoise = round(max([prop.area for prop in properties])) - 2
        tumorb = morphology.remove_small_objects(tumorb, thrNoise,
                                                 connectivity=2)  # connectivity=2 ensure diagnoal pixels are neightbors

    til_in_tumor = np.logical_and(tumorb, til_mask)
    tumor_no_til = np.logical_and(tumorb, np.invert(til_in_tumor))

    tumor_circle_mask, tumor_circle_mask2 = iterate_circles(tumor_no_til)

    tumor_no_til_c=np.logical_and(tumor_no_til,np.invert(tumor_circle_mask2))
    tumor_no_til_cs=np.zeros_like(tumor_no_til_c)
    tumor_no_til_cs[0:2:tumor_no_til.shape[0],0:2:tumor_no_til.shape[1]]=tumor_no_til_c[0:2:tumor_no_til.shape[0],0:2:tumor_no_til.shape[1]]

    im = Image.fromarray((tumor_circle_mask2 * 127).astype(np.uint8))
    im.save('../../../data_history/debugs/' + file_img.split('/')[-2] + '_' + file_img.split('/')[-1] + '.png')




if __name__=='__main__':
    ## whole slide image path
    imagePath = ['../../data/kang_colon_slide/181119/',
                 '../../data/kang_colon_slide/181211/']
                 #'../../../data/kang_colon_slide/Kang_MSI_WSI_2019_10_07/']

    ## tumor prediction mask
    tumorPath = ['../../data/kang_colon_data/td_models/predictions_kang/dl_model_v01/181119_low/',
                 '../../data/kang_colon_data/td_models/predictions_kang/dl_model_v01/181211_low/']
                 #'../../../data/kang_colon_data/predictions_tumor/dl_model_v01/Kang_MSI_WSI_2019_10_07_low/']

    ## til prediction mask
    tilPath= ['../../data/pan_cancer_tils/data_yonsei_v01_pred/181119/',
              '../../data/pan_cancer_tils/data_yonsei_v01_pred/181211/']
              #'../../../data/pan_cancer_tils/data_yonsei_v01_pred//Kang_MSI_WSI_2019_10_07/']
    thr=0.5 # threshold on tumor prediction map
    mag = 0.078125*2
    for i in range(len(imagePath)):
        t_imagePath=imagePath[i]
        t_tumorPath=tumorPath[i]
        t_tilPath=tilPath[i]
        wsis=sorted(os.listdir(t_imagePath))
        for img_name in wsis[5:]:
            if '.mrxs' in img_name:
                file_img=t_imagePath+img_name
                file_tumor=t_tumorPath+img_name+'.png'
                file_til=t_tilPath+img_name.split('.')[0]+'_color.png'

                tumor_til_analysis_v02(file_img, file_tumor, file_til, thr, mag)
