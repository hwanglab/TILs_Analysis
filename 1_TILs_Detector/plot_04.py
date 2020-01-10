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

msi_path=['../../../data/kang_colon_data/msi_binary/Kang_COLON_WSI_181119_tiled_normalized_v1/',
          '../../../data/kang_colon_data/msi_binary/Kang_COLON_WSI_181211_tiled_normalized_v1/',
          '../../../data/kang_colon_data/msi_binary/Kang_MSI_WSI_2019_10_07_tiled_normalized_v1/']

til_path=['../../../data/pan_cancer_tils/data_yonsei_v01_pred/181119/',
          '../../../data/pan_cancer_tils/data_yonsei_v01_pred/181211/',
          '../../../data/pan_cancer_tils/data_yonsei_v01_pred/Kang_MSI_WSI_2019_10_07/']

tumor_path=['../../../data/kang_colon_data/predictions_tumor/dl_model_v01/181119_low/',
            '../../../data/kang_colon_data/predictions_tumor/dl_model_v01/181211_low/',
            '../../../data/kang_colon_data/predictions_tumor/dl_model_v01/Kang_MSI_WSI_2019_10_07_low/']

out_path=['../../../data_history/181119/',
          '../../../data_history/181211/',
          '../../../data_history/Kang_MSI_WSI_2019_10_07/']

if __name__=='__main__':
    thr=0.5
    thrNoise=64
    for ind,msi in enumerate(msi_path):
        imgs=glob.glob(msi+'*.png')
        temp_til=til_path[ind]
        temp_tumor=tumor_path[ind]
        for im,img in enumerate(imgs):
            img_m=plt.imread(img)
            msi_m=img_m[:,:,0]

            til_m=plt.imread(temp_til+img.split('\\')[-1].split('.')[0]+'_color.png') # run on win10 system
            msi_m = skimage.transform.resize(msi_m, til_m.shape[0:2],
                                                  order=0)  # 0 nearest-neighbor, 1: bi-linear (default)

            tumor_m=plt.imread(temp_tumor+img.split('\\')[-1].split('.')[0]+'.mrxs.png')
            tumor_mask = skimage.transform.resize(tumor_m, til_m.shape[0:2], order=1)
            bw_mask0 = tumor_mask > thr
            bw_mask0 = morphology.remove_small_objects(bw_mask0, thrNoise)
            bw_mask0 = morphology.remove_small_holes(bw_mask0, thrNoise)

            til_msi=np.logical_or(msi_m,til_m[:,:,0])
            tum_m=np.logical_and(bw_mask0,np.invert(til_msi))

            til_m[:,:,1]=msi_m
            til_m[:,:,2]=tum_m

            imsave(out_path[ind]+img.split('\\')[-1], til_m)


