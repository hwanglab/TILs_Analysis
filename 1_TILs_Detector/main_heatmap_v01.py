'''
generate heatmaps
author: HONGMING XU
CCF
qeutions: mxu@ualberta.ca
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

global t_g
t_g = 0.2  # a key threshold, attentions?

def parallel_filling(i,X,Y,img_name,Stride,File):
    Slide = openslide.OpenSlide(File)

    for j in range(X.shape[1] - 1):
        Tile = Slide.read_region((int(X[i, j]), int(Y[i, j])), 0, (Stride[0], Stride[1]))
        Tile = np.asarray(Tile)
        Tile = Tile[:, :, :3]
        bn = np.sum(Tile[:, :, 0] < 5) + np.sum(Tile[:, :, 0] > 250)
        if (np.std(Tile[:, :, 0]) + np.std(Tile[:, :, 1]) + np.std(Tile[:, :, 2])) / 3 > 18 and bn < Stride[0] * Stride[
            1] * 0.3:
            tile_name = img_name.split('.')[0] + '_' + str(X[i, j]) + '_' + str(Y[i, j]) + '_' + str(
                Stride[0]) + '_' + str(Stride[1]) + '_' + '.png'
            ind = np.where(df_g['Name'] == tile_name)
            index = int(ind[0])
            pred_g_g[i, j, :] = list(df_g['Pred'])[index] * 255
            if df_g['Pred'][index] > t_g:
                pred_c_g[i, j, 0] = 255
            else:
                pred_c_g[i, j, 2] = 255

def wsi_tiling(File,temp_predPath, dest_imagePath,img_name,Tile_size):
    since = time.time()

    df=pd.read_excel(temp_predPath+img_name.split('.')[0]+'.xlsx')
    # open image
    Slide = openslide.OpenSlide(File)

    xr = float(Slide.properties['openslide.mpp-x'])  # pixel resolution at x direction
    yr = float(Slide.properties['openslide.mpp-y'])  # pixel resolution at y direction
    # generate X, Y coordinates for tiling
    Stride = [round(Tile_size[0] / xr), round(Tile_size[1] / yr)]
    Dims = Slide.level_dimensions
    X = np.arange(0, Dims[0][0] + 1, Stride[0])
    Y = np.arange(0, Dims[0][1] + 1, Stride[1])
    X, Y = np.meshgrid(X, Y)

    # MappingMag=2.5
    # Level, Tout, Factor = wsi_coarse_level(Slide, MappingMag, Stride)
    #
    # # get width, height of image at low-res reading magnification
    # lrHeight = Slide.level_dimensions[Level][1]
    # lrWidth = Slide.level_dimensions[Level][0]
    #
    # # read in whole slide at low magnification
    # LR = Slide.read_region((0, 0), Level, (lrWidth, lrHeight))
    #
    # # convert to numpy array and strip alpha channel
    # LR = np.asarray(LR)
    # LR = LR[:, :, :3]


    pred_c=np.zeros((X.shape[0]-1,X.shape[1]-1,3),'uint8')
    pred_g=np.zeros((X.shape[0]-1,X.shape[1]-1,3),'float')

    global pred_c_g
    pred_c_g=pred_c
    global pred_g_g
    pred_g_g=pred_g
    global df_g
    df_g=df

    # parallel-running
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
         for _ in executor.map(parallel_filling, list(range(X.shape[0]-1)), repeat(X), repeat(Y),repeat(img_name),repeat(Stride),repeat(File)):
             pass

    # for i in range(X.shape[0] - 1):
    #     for j in range(X.shape[1] - 1):
    #             Tile = Slide.read_region((int(X[i, j]), int(Y[i, j])), 0, (Stride[0], Stride[1]))
    #             Tile = np.asarray(Tile)
    #             Tile = Tile[:, :, :3]
    #             bn=np.sum(Tile[:, :, 0] < 5) + np.sum(Tile[:, :, 0] > 250)
    #             if (np.std(Tile[:,:,0])+np.std(Tile[:,:,1])+np.std(Tile[:,:,2]))/3>18 and bn<Stride[0]*Stride[1]*0.3:
    #                 tile_name=img_name.split('.')[0]+'_'+str(X[i,j])+'_'+str(Y[i,j])+'_'+str(Stride[0])+'_'+str(Stride[1])+'_'+'.png'
    #                 ind=np.where(df['Name']==tile_name)
    #                 index=int(ind[0])
    #                 pred_g[i,j,:]=list(df['Pred'])[index]*255
    #                 if df['Pred'][index]>t:
    #                     pred_c[i,j,0]=255
    #                 else:
    #                     pred_c[i,j,2]=255
    #
    img1 = Image.fromarray(pred_c_g)
    img2 = Image.fromarray(pred_g_g.astype('uint8'))
    img1.save(dest_imagePath+img_name.split('.')[0]+'_'+'color.png')
    img2.save(dest_imagePath + img_name.split('.')[0] + '_' + 'gray.png')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

if __name__=='__main__':
    imagePath=['../../../data/kang_colon_slide/181119/']
    predPath=['../../../data/pan_cancer_tils/data_yonsei_v01_pred/181119_pred/']
    destPath=['../../../data/pan_cancer_tils/data_yonsei_v01_pred/181119/']
    #tileSize=[50,50] # micro-meters
    tileSize=[112,112] # micro-meters
    for i in range(len(imagePath)):
        temp_imagePath = imagePath[i]
        temp_predPath = predPath[i]
        dest_imagePath = destPath[i]
        wsis = sorted(os.listdir(temp_imagePath))
        for img_name in wsis[:15]:
            if '.mrxs' in img_name:
                file=temp_imagePath+img_name
                wsi_tiling(file, temp_predPath, dest_imagePath,img_name, tileSize)