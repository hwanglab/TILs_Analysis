'''
main function to analyze tumor-til-maps
author: Hongming Xu, 2020, CCF
email: mxu@ualberta.ca

purpose: analyze tumor invasive margin regions

input:
    wsi image
    tumor prediction map
    til prediction map
output:
    feature vector for the corresponding wsi

        feat1: number of tils inside tumor
        feat2: number of tils inside invasive margin (200,300,400,500)
        ....
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
import sys

relpath='../../' # the relative path other folders that are used
sys.path.insert(0,relpath+'xhm_deep_learning/functions')
from wsi_preprocess_mask import wsi_preprocess_mask_v01
from wsi_coarse_level import wsi_coarse_read

def overlap_contour(LR2,contours,color):
    for n,contour in enumerate(contours):
        r=np.round(contour[:,0]).astype(int)
        c=np.round(contour[:,1]).astype(int)
        LR2[r,c,0]=color[0]
        LR2[r,c,1]=color[1]
        LR2[r,c,2]=color[2]

    return LR2

def tumor_til_analysis(file_img,file_tumor,file_til,thr,mag):
    # open slide
    Slide=openslide.OpenSlide(file_img)
    LR, Objective, pxy=wsi_coarse_read(Slide,mag) # at 2.5 magnification

    # open til mask
    til_map=plt.imread(file_til)
    til_mask=til_map[:,:,0]

    tissue_mask = np.logical_or(til_mask,til_map[:,:,2])
    til_mask=transform.resize(til_mask,LR.shape[0:2],order=0) # order=0 nearest-neighbor
    tissue_mask=transform.resize(tissue_mask,LR.shape[0:2],order=0)

    # open tumor mask
    tumor_mask=plt.imread(file_tumor)
    if tumor_mask.shape!=LR.shape[0:2]:
        tumor_mask=transform.resize(tumor_mask,LR.shape[0:2],order=1) # order=0 nearest-neighbor
    tumorb=wsi_preprocess_mask_v01(tumor_mask,thr)

    # locate tumor invasive margin regions
    pr_mag=(Objective/mag)*pxy[0] # pixel resolution at mag, assume pxy[0]=pxy[1] in therory
    inv_mar=[200,300,400,500] # in terms of micro meters
    pp_mar=[temp/pr_mag for temp in inv_mar]
    # only consider the largest tumor region
    mask_label = measure.label(tumorb, neighbors=8,background=0)
    properties = measure.regionprops(mask_label)
    if len(properties)>1:
        thrNoise=round(max([prop.area for prop in properties]))-2
        tumorb = morphology.remove_small_objects(tumorb, thrNoise,connectivity=2) # connectivity=2 ensure diagnoal pixels are neightbors

    contours=measure.find_contours(tumorb,0.5)
    LR2=LR.copy()
    LR2=overlap_contour(LR2,contours,[255,255,0])

    feat['feat0'].append(np.sum(np.logical_and(tumorb,til_mask))/np.sum(tumorb))

    color=[[0,255,0],[0,0,255],[0,255,255],[128,128,0]]
    for ind,k in enumerate(pp_mar):
        selem=np.ones((round(k*2+1),round(k*2+1)),dtype=np.uint8)
        tumorb_inv=morphology.binary_dilation(tumorb,selem)
        inv_mask=np.logical_xor(tumorb,tumorb_inv)
        inv_mask=np.logical_and(inv_mask,tissue_mask)
        til_den=np.sum(np.logical_and(inv_mask,til_mask))/np.sum(inv_mask)
        feat['feat'+str(ind+1)].append(til_den)

        if debug==True:
            temp_inv_mask=np.logical_or(inv_mask,tumorb)
            contours = measure.find_contours(temp_inv_mask, 0.5)
            LR2=overlap_contour(LR2,contours,color[ind])

    if debug==True:
        im=Image.fromarray(LR2)
        im.save(relpath+'data/pan_cancer_tils/debug/' + file_img.split('/')[-2]+'_'+file_img.split('/')[-1] + '.png')

    ## plot figure and contours on figures
    # f=plt.figure()
    # plt.imshow(LR)
    # plt.contour(tumorb, [0.5], colors=['yellow'],linewidths=0.5)
    # colors=['red','blue','green','cyan']
    # for ind,k in enumerate(pp_mar):
    #     selem=np.ones((round(k*2+1),round(k*2+1)),dtype=np.uint8)
    #     tumorb_inv=morphology.binary_dilation(tumorb,selem)
    #     inv_mask=np.logical_xor(tumorb,tumorb_inv)
    #     inv_mask=np.logical_and(inv_mask,tissue_mask)
    #     plt.contour(np.logical_or(inv_mask,tumorb), [0.5], colors=colors[ind],linewidths=0.5)
    #
    # plt.savefig('../../../data/pan_cancer_tils/debug/' + file_img.split('/')[-2]+'_',file_img.split('/')[-1] + '.png')
    # f.clear()
    # plt.close()



if __name__=='__main__':
    ## whole slide image path
    imagePath = [relpath+'data/kang_colon_slide/181119/',
                 relpath+'data/kang_colon_slide/181211/']
                 #'../../../data/kang_colon_slide/Kang_MSI_WSI_2019_10_07/']

    ## tumor prediction mask
    tumorPath = [relpath+'data/kang_colon_data/td_models/predictions_kang/dl_model_v01/181119/',
                 relpath+'data/kang_colon_data/td_models/predictions_kang/dl_model_v01/181211/']
                 #'../../../data/kang_colon_data/predictions_tumor/dl_model_v01/Kang_MSI_WSI_2019_10_07/']

    ## til prediction mask
    tilPath= [relpath+'data/pan_cancer_tils/data_yonsei_v01_pred/181119/',
              relpath+'data/pan_cancer_tils/data_yonsei_v01_pred/181211/']
              #'../../../data/pan_cancer_tils/data_yonsei_v01_pred//Kang_MSI_WSI_2019_10_07/']

    global debug
    debug=False

    patient_id=[]
    global feat
    feat={}
    feat['feat0']=[]
    feat['feat1']=[]
    feat['feat2']=[]
    feat['feat3']=[]
    feat['feat4']=[]


    thr=0.5 # threshold on tumor prediction map
    mag = 0.078125*2
    for i in range(len(imagePath)):
        t_imagePath=imagePath[i]
        t_tumorPath=tumorPath[i]
        t_tilPath=tilPath[i]
        wsis=sorted(os.listdir(t_imagePath))
        for img_name in wsis:
            if '.mrxs' in img_name:
                file_img=t_imagePath+img_name
                file_tumor=t_tumorPath+img_name+'.png'
                file_til=t_tilPath+img_name.split('.')[0]+'_color.png'

                patient_id.append(t_imagePath.split('/')[-2]+'_'+img_name.split('.')[0])

                tumor_til_analysis(file_img,file_tumor,file_til,thr,mag)

    save_feat=True
    if save_feat==True:
        data={'patient id': patient_id}
        data.update(feat)
        df=pd.DataFrame(data)

        feat_out=relpath+'data/pan_cancer_tils/feat_tils/yonsei_colon/'+'til_density.xlsx'
        df.to_excel(feat_out)
