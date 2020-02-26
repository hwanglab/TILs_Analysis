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

def tumor_til_analysis(file_img,file_tumor,file_til,thr,mag,ignore_small_inv=False,pp=0):
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

    til_map2 = transform.resize(til_map, LR.shape, order=0)
    til_map2=(til_map2*255).astype('uint8')
    til_map2 = overlap_contour(til_map2,contours,[255,255,255])
    color = [[0, 255, 0], [0, 0, 255], [0, 255, 255], [128, 128, 0]]


    if ignore_small_inv==True:

        for ind, k in enumerate(pp_mar):
            selem = np.ones((round(k * 2 + 1), round(k * 2 + 1)), dtype=np.uint8)
            tumorb_inv = morphology.binary_dilation(tumorb, selem)
            inv_mask = np.logical_xor(tumorb, tumorb_inv)
            size_org=np.sum(inv_mask)
            inv_mask = np.logical_and(inv_mask, tissue_mask)
            size_ff=np.sum(inv_mask)

            if size_ff/size_org<pp:
                #and ind==0:  #for simplicity, at current stage, we only consider the size of 200um to filter out tumors with im regions
                print(f"{file_img} has no invasive margins!!")
                for ind2 in range(ind,len(pp_mar)):
                    feat['feat' + str(ind2 + 1)].append(np.nan)

                break

                #return False
            else:
                til_den = np.sum(np.logical_and(inv_mask, til_mask)) / np.sum(inv_mask)
                feat['feat' + str(ind + 1)].append(til_den)

                if debug == True:
                    temp_inv_mask = np.logical_or(inv_mask, tumorb)
                    contours = measure.find_contours(temp_inv_mask, 0.5)
                    LR2 = overlap_contour(LR2, contours, color[ind])

                    til_map2 = overlap_contour(til_map2, contours, [255, 255, 255])

        feat['feat0'].append(np.sum(np.logical_and(tumorb, til_mask)) / np.sum(tumorb))

        selem_inside=np.ones((round(pp_mar[0] * 2 + 1), round(pp_mar[0] * 2 + 1)), dtype=np.uint8)
        tumorb_center=morphology.binary_erosion(tumorb,selem_inside)
        tumorb_inv_inverse=np.logical_xor(tumorb,tumorb_center)
        feat['feat5'].append(np.sum(np.logical_and(tumorb_inv_inverse,til_mask))/np.sum(tumorb_inv_inverse))
        feat['feat6'].append(np.sum(np.logical_and(tumorb_center,til_mask))/np.sum(tumorb_center))

        contours=measure.find_contours(tumorb_center,0.5)
        LR2=overlap_contour(LR2,contours,[255,255,255])

    else:
        feat['feat0'].append(np.sum(np.logical_and(tumorb,til_mask))/np.sum(tumorb))

        selem_inside = np.ones((round(pp_mar[0] * 2 + 1), round(pp_mar[0] * 2 + 1)), dtype=np.uint8)
        tumorb_center = morphology.binary_erosion(tumorb, selem_inside)
        tumorb_inv_inverse = np.logical_xor(tumorb, tumorb_center)
        feat['feat5'].append(np.sum(np.logical_and(tumorb_inv_inverse, til_mask)) / np.sum(tumorb_inv_inverse))
        feat['feat6'].append(np.sum(np.logical_and(tumorb_center, til_mask)) / np.sum(tumorb_center))

        contours = measure.find_contours(tumorb_center, 0.5)
        LR2 = overlap_contour(LR2, contours, [255, 255, 255])

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

                til_map2 = overlap_contour(til_map2, contours, [255,255,255])


    if debug==True:
        im=Image.fromarray(LR2)
        if yonsei_colon==True:
            im.save(relpath+'data/pan_cancer_tils/contours_im/yonsei_im/' + file_img.split('/')[-2]+'_'+file_img.split('/')[-1] + '.png')
        elif tcga_coad==True:
            im.save(relpath + 'data/pan_cancer_tils/contours_im/tcga_coad/' + file_img.split('/')[-1] + '.png')
        else:
            raise RuntimeError('incorrect selection....')

    #return True

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


yonsei_colon=False
tcga_coad=True
if __name__=='__main__':

    if yonsei_colon==True:
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

        feat_out = relpath + 'data/pan_cancer_tils/feat_tils/yonsei_colon/' + 'til_density.xlsx'
        thr = 0.5  # threshold on tumor prediction map
        mag = 0.078125 * 2
        wsi_type='.mrxs'

        ignore_small_inv=False ## visually all tumors have invasive margins
    elif tcga_coad==True:
        imagePath=[relpath+'data/tcga_coad_slide/tcga_coad/quality_a1/',
                   relpath+'data/tcga_coad_slide/tcga_coad/quality_a2/',
                   relpath+'data/tcga_coad_slide/tcga_coad/quality_b/',
                   relpath+'data/tcga_coad_slide/tcga_coad/quality_uncertain/']
        tumorPath=[relpath+'data/tcga_coad_read_data/coad_tumor_preds/resnet18_tcga_v2_tils/',
                   relpath+'data/tcga_coad_read_data/coad_tumor_preds/resnet18_tcga_v2_tils/',
                   relpath+'data/tcga_coad_read_data/coad_tumor_preds/resnet18_tcga_v2_tils/',
                   relpath+'data/tcga_coad_read_data/coad_tumor_preds/resnet18_tcga_v2_tils/']
        tilPath=[relpath+'data/tcga_coad_read_data/coad_read_tils_preds/pred_maps_0.5/',
                 relpath+'data/tcga_coad_read_data/coad_read_tils_preds/pred_maps_0.5/',
                 relpath+'data/tcga_coad_read_data/coad_read_tils_preds/pred_maps_0.5/',
                 relpath+'data/tcga_coad_read_data/coad_read_tils_preds/pred_maps_0.5/']
        feat_out0 = relpath + 'data/pan_cancer_tils/feat_tils/tcga_coad/'
        thr = 0.5  # threshold on tumor prediction map
        mag = 0.625
        wsi_type='.svs'

        ignore_small_inv = True # visually some tumors do not have invasive margins
        inv_p=[0.4,0.5,0.6,0.7]

        # read excel table
        df=pd.read_excel(relpath+'data/tcga_coad_slide/TCGA-COAD_patient_info.xlsx')
    else:
        raise RuntimeError("incorrect dataset switches....see dataset selection!!!")

    global debug
    debug = False
    for pp in inv_p:
        feat_out=feat_out0+ 'til_density' + str(pp)+'.xlsx'

        patient_id=[]
        global feat
        feat={}
        feat['feat0']=[] # tils in whole tumor
        feat['feat1']=[] # tils in 200um im
        feat['feat2']=[] # tils in 300um im
        feat['feat3']=[] # tils in 400um im
        feat['feat4']=[] # tils in 500um im
        feat['feat5']=[] # tils in inverse 200um im
        feat['feat6']=[] # tils in tumor center

        for i in range(len(imagePath)):
            t_imagePath=imagePath[i]
            t_tumorPath=tumorPath[i]
            t_tilPath=tilPath[i]
            wsis=sorted(os.listdir(t_imagePath))
            for img_name in wsis:
                if wsi_type in img_name:
                    if yonsei_colon==True:
                        file_img=t_imagePath+img_name
                        file_tumor=t_tumorPath+img_name+'.png'
                        file_til=t_tilPath+img_name.split('.')[0]+'_color.png'
                        temp_pid=t_imagePath.split('/')[-2]+'_'+img_name.split('.')[0]

                    elif tcga_coad==True:
                        file_img = t_imagePath + img_name
                        file_tumor = t_tumorPath + img_name[0:23] + '.png'
                        file_til = t_tilPath + img_name[0:23] + '_color.png'
                        try:
                            ind=df['bcr_patient_barcode'].tolist().index(img_name[0:12])
                            if isinstance(df['stage_event_pathologic_stage'][ind], str):
                                if df['stage_event_pathologic_stage'][ind][6:8]=='II' or df['stage_event_pathologic_stage'][ind][6:9]=='III':
                                    temp_pid=img_name[0:23]
                                else:
                                    print(f"{img_name} has stage {df['stage_event_pathologic_stage'][ind]}")
                                    continue
                            else:
                                print(f"{img_name} has no stage info {df['stage_event_pathologic_stage'][ind]}")
                                continue
                        except:
                            print(f"{img_name} not in the excel file patient info")
                            continue
                    else:
                        raise RuntimeError('undefined selection~~~~~~~')


                    tumor_til_analysis(file_img,file_tumor,file_til,thr,mag,ignore_small_inv,pp)

                    #if pat_add==True:
                    patient_id.append(temp_pid)
                    #else:
                    #    print(f"skip patient {temp_pid} due to too small invarive margin regions..")


        ## only first run--save features
        save_feat=True
        if save_feat==True:
            data={'patient id': patient_id}
            data.update(feat)
            df2=pd.DataFrame(data)

            df2.to_excel(feat_out)
