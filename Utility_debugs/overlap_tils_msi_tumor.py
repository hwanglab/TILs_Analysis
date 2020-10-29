'''
purpose: overlap msi_tils_tumor map together

author: Hongming Xu, CCF, 2020
email: mxu@ualberta.ca
'''
import glob
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
from skimage.io import imsave
from skimage import morphology
import shutil
import pandas as pd

rela_path='../../../'

yonsei_colon=False
immuno_gastric=False
tcga_stad=True

if yonsei_colon==True:
    msi_path=[rela_path+'data/pan_cancer_tils/data_yonsei_v01_pred/Yonsei_colon_MSI_predictions/Kang_COLON_WSI_181119_tiled_normalized_v1/',
              rela_path+'data/pan_cancer_tils/data_yonsei_v01_pred/Yonsei_colon_MSI_predictions//Kang_COLON_WSI_181211_tiled_normalized_v1/',
              rela_path+'data/pan_cancer_tils/data_yonsei_v01_pred/Yonsei_colon_MSI_predictions//Kang_MSI_WSI_2019_10_07_tiled_normalized_v1/']

    til_path=[rela_path+'data/pan_cancer_tils/data_yonsei_v01_pred/pred_images0.4/181119/',
              rela_path+'data/pan_cancer_tils/data_yonsei_v01_pred/pred_images0.4/181211/',
              rela_path+'data/pan_cancer_tils/data_yonsei_v01_pred/pred_images0.4/Kang_MSI_WSI_2019_10_07/']

    tumor_path=[rela_path+'data/kang_colon_data/td_models/predictions_kang/dl_model_v01/181119_low/',
                rela_path+'data/kang_colon_data/td_models/predictions_kang/dl_model_v01/181211_low/',
                rela_path+'data/kang_colon_data/td_models/predictions_kang/dl_model_v01/Kang_MSI_WSI_2019_10_07_low/']

    out_path=[rela_path+'data_history/181119/',
              rela_path+'data_history/181211/',
              rela_path+'data_history/Kang_MSI_WSI_2019_10_07/']

    flag='*.png'
elif immuno_gastric==True:
    msi_path=[rela_path+'data/stomach_cancer_immunotherapy/msi_maps/predictions_GC_SM2_stmary_analysis_v3/',
              rela_path+'data/stomach_cancer_immunotherapy/msi_maps/predictions_Stomach_Cancer_Stage4_Immunotherapy_analysis_v3/',
              rela_path+'data/stomach_cancer_immunotherapy/msi_maps/predictions_Stomach_Immunotherapy_stmary_analysis_v3/']
    tumor_path=[rela_path+'data/stomach_cancer_immunotherapy/msi_maps/tumor_maps_v2/GC_SM2_stmary/',
                rela_path+'data/stomach_cancer_immunotherapy/msi_maps/tumor_maps_v2/Stomach_Cancer_Stage4_Immunotherapy/',
                rela_path+'data/stomach_cancer_immunotherapy/msi_maps/tumor_maps_v2/Stomach_Immunotherapy_stmary/']
    til_path=[rela_path+'data/stomach_cancer_immunotherapy/tils_maps/GC_SM2_stmary/',
              rela_path+'data/stomach_cancer_immunotherapy/tils_maps/Stomach_Cancer_Stage4_Immunotherapy/',
              rela_path+'data/stomach_cancer_immunotherapy/tils_maps/Stomach_Immunotherapy_stmary/']
    out_path=[rela_path+'data/stomach_cancer_immunotherapy/msi_til_maps_updated/GC_SM2_stmary/',
              rela_path+'data/stomach_cancer_immunotherapy/msi_til_maps_updated/Stomach_Cancer_Stage4_Immunotherapy/',
              rela_path+'data/stomach_cancer_immunotherapy/msi_til_maps_updated/Stomach_Immunotherapy_stmary/']
    flag = '*.png'
elif tcga_stad==True:
    msi_path = [rela_path + 'data/tcga_stad_slide/predictions_TCGA_STAD_analysis_binary_v1 (2)/']
    tumor_path = [rela_path + 'data/tcga_stad_slide/tumor_tiles/']
    til_path = [rela_path + 'data/tcga_stad_slide/til_maps/wsis/']
    out_path = [rela_path + 'data/tcga_stad_slide/tumor_msi_tils_maps/']
    flag = '*.png'
else:
    raise RuntimeError('undefined dataset!!!!')

def binary_overlap():
    thr = 0.5
    thrNoise = 64
    for ind, msi in enumerate(msi_path):
        imgs = glob.glob(msi + '*.png')
        temp_til = til_path[ind]
        temp_tumor = tumor_path[ind]
        for im, img in enumerate(imgs):
            img_m = plt.imread(img)
            msi_m = img_m[:, :, 0]

            til_m = plt.imread(temp_til + img.split('\\')[-1].split('.')[0] + '_color.png')  # run on win10 system
            msi_m = skimage.transform.resize(msi_m, til_m.shape[0:2],
                                             order=0)  # 0 nearest-neighbor, 1: bi-linear (default)

            tumor_m = plt.imread(temp_tumor + img.split('\\')[-1].split('.')[0] + '.mrxs.png')
            tumor_mask = skimage.transform.resize(tumor_m, til_m.shape[0:2], order=1)
            bw_mask0 = tumor_mask > thr
            bw_mask0 = morphology.remove_small_objects(bw_mask0, thrNoise)
            bw_mask0 = morphology.remove_small_holes(bw_mask0, thrNoise)

            til_msi = np.logical_or(msi_m, til_m[:, :, 0])
            tum_m = np.logical_and(bw_mask0, np.invert(til_msi))

            til_m[:, :, 1] = msi_m
            til_m[:, :, 2] = tum_m

            imsave(out_path[ind] + img.split('\\')[-1], til_m)

def gray_overlap():
    til_density=[]
    img_id=[]
    til_density_v2=[]

    for ind, msi in enumerate(msi_path):
        imgs = glob.glob(msi + '*.png')
        temp_til = til_path[ind]
        temp_tumor = tumor_path[ind]
        for im, img in enumerate(imgs):
            img_m = plt.imread(img)
            msi_m = img_m[:, :, 0]

            if immuno_gastric==True:
                img_name=img.split('\\')[-1].split('.')[0][:-31]
                til_m = plt.imread(
                    temp_til + img_name + '_gray.png')  # run on win10 system
                # tumor_m = plt.imread(temp_tumor + img.split('\\')[-1].split('.')[0] + '.mrxs.png')
                tumor_m = plt.imread(
                    temp_tumor + img_name + '_Tumor_detection_heatmap.png')
            elif tcga_stad==True:
                img_name=img.split('\\')[-1].split('.')[0]

                try:
                    til_m = plt.imread(
                        temp_til + img_name + '_gray.png')  # run on win10 system
                    tumor_m = plt.imread(
                        temp_tumor + img_name + '_gray.png')
                except:
                    print('wsi=%s' % img_name)
                    continue
            else:
                raise RuntimeError('undefined option...')

            msi_m = skimage.transform.resize(msi_m, til_m.shape[0:2],
                                             order=1)  # 0 nearest-neighbor, 1: bi-linear (default)


            tumor_mask = skimage.transform.resize(tumor_m, til_m.shape[0:2], order=1)

            # overlap probability maps: R-tils, G-msi, B-tumor
            tum_m=tumor_mask[:, :, 0]
            #til_msi = np.logical_or(msi_m, til_m[:, :, 0])
            #tum_m = np.logical_and(bw_mask0, np.invert(til_msi))
            til_m[:, :, 1] = msi_m
            til_m[:, :, 2] = tum_m

            # figure 1: save probabiity maps
            #imsave(out_path[ind] + img.split('\\')[-1].split('.')[0][:-31]+'_probability.png', til_m)

           # overlap binary maps
            tumor_mask=(til_m[:, :, 2] >= 0.5)
            til_m2=np.zeros(til_m.shape)
            til_m2[:,:,0] = np.logical_and(tumor_mask,(til_m[:,:,0]>=0.5))
            til_m2[:,:,1] = np.logical_and(tumor_mask,(til_m[:,:,1]>=0.5))
            #til_m2[:,:,2] = np.zeros(til_m2[:,:,2].shape)
            til_m2[:, :, 2] = (til_m[:, :, 2] >= 0.5)

            # figure2: save binary maps
            imsave(out_path[ind] + img_name + '_binary.png', til_m2)

            # figure3: remove msi-h in non-tumor regions
            til_m[:,:,1]=np.multiply(til_m[:,:,1],tumor_mask)
            imsave(out_path[ind] + img_name + '_probability.png', til_m)

            if np.sum(tumor_mask)==0:
                print(img_name)

            # til density over whole tumor regions
            til_density.append(np.sum(til_m2[:,:,0])/np.sum(tumor_mask))
            img_id.append(img_name)

            # til density & msi-h over whole tumor regions
            til_msih=np.logical_and(til_m2[:,:,0] ,til_m2[:,:,1])
            til_density_v2.append(np.sum(til_msih) / np.sum(tumor_mask))


    # save excel file
    data={'img_id':img_id,'til_density':til_density,'til_density_v2':til_density_v2}
    df=pd.DataFrame(data)

    if immuno_gastric==True:
        df.to_csv(rela_path+'data/stomach_cancer_immunotherapy/'+'til_density_of_tumor.csv')
    elif tcga_stad==True:
        df.to_csv(out_path[0] + 'til_density_of_tumor.csv')
    else:
        pass

def move_files():

    if immuno_gastric==True:
        source = [rela_path+'data/stomach_cancer_immunotherapy/msi_maps/predictions_GC_SM2_stmary_analysis_v3/',
                  rela_path+'data/stomach_cancer_immunotherapy/msi_maps/predictions_Stomach_Cancer_Stage4_Immunotherapy_analysis_v3/',
                  rela_path+'data/stomach_cancer_immunotherapy/msi_maps/predictions_Stomach_Immunotherapy_stmary_analysis_v3/']

        dest = [rela_path+'data/stomach_cancer_immunotherapy/msi_maps/tumor_maps_v2/GC_SM2_stmary/',
                rela_path+'data/stomach_cancer_immunotherapy/msi_maps/tumor_maps_v2/Stomach_Cancer_Stage4_Immunotherapy/',
                rela_path+'data/stomach_cancer_immunotherapy/msi_maps/tumor_maps_v2/Stomach_Immunotherapy_stmary/']
        flag='*predicted_tumorous_regions.png'
    elif tcga_stad==True:
        source =  [rela_path + 'data/tcga_stad_slide/predictions_TCGA_STAD_analysis_binary_v1 (2)/']
        dest =  [rela_path + 'data/tcga_stad_slide/binary_maps/']
        flag='*Thresholded_at_0.5_TCGA-trained_model_MSI_heatmap_of_predicted_tumorous_regions.png'
    else:
        raise RuntimeError('undefined option....')

    for ind, msi in enumerate(source):
        imgs = glob.glob(msi + flag)
        dest_path=dest[ind]
        for im, img in enumerate(imgs):
            shutil.move(img,dest_path)
if __name__=='__main__':
    # optional preparation step depending image provided by Isaiah
    #move_files() # - only run in the first time for processing

    gray_overlap() # overlap function


