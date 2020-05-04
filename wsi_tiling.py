'''
divide whole slide image into tiles and save the tiles into local disk

background tiles are removed based on some criteira (see code)

purpose: tiling the whole slide images
author: HONGMING XU, CCF
email: mxu@ualberta.ca
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
from tqdm import tqdm
import pandas as pd
from skimage import transform

rela_path='../../'
import sys
sys.path.insert(0,rela_path+'xhm_deep_learning/functions')
from MacenkoNormalizer import MacenkoNormalizer

def wsi_coarse_level(Slide,Magnification,Stride,tol=0.02):
    # get slide dimensions, zoom levels, and objective information
    Factors = Slide.level_downsamples
    Objective = float(Slide.properties[
                          openslide.PROPERTY_NAME_OBJECTIVE_POWER])

    # determine if desired magnification is avilable in file
    Available = tuple(Objective / x for x in Factors)
    Mismatch = tuple(x - Magnification for x in Available)
    AbsMismatch = tuple(abs(x) for x in Mismatch)
    if min(AbsMismatch) <= tol:
        Level = int(AbsMismatch.index(min(AbsMismatch)))
        Factor = 1
    else:
        if min(Mismatch) < 0:  # determine is there is magnifications below 2.5x
            # pick next lower level, upsample
            Level = int(min([i for (i, val) in enumerate(Mismatch) if val < 0]))
        else:
            # pick next higher level, downsample
            Level = int(max([i for (i, val) in enumerate(Mismatch) if val > 0]))

        Factor = Magnification / Available[Level]

    # translate parameters of input tiling schedule into new schedule
    Tout = [round(Stride[0]*Magnification/Objective), round(Stride[0]*Magnification/Objective)]


    return Level,Tout,Factor

def parallel_tiling(i,X,Y,dest_imagePath,img_name,Stride,File,color_norm):
    Slide = openslide.OpenSlide(File)

    for j in range(X.shape[1] - 1):
        Tile = Slide.read_region((int(X[i, j]), int(Y[i, j])), 0, (Stride[0], Stride[1]))
        Tile = np.asarray(Tile)
        Tile = Tile[:, :, :3]
        bn = np.sum(Tile[:, :, 0] < 5) + np.sum(np.mean(Tile,axis=2) > 245)
        if (np.std(Tile[:, :, 0]) + np.std(Tile[:, :, 1]) + np.std(Tile[:, :, 2])) / 3 > 18 and bn < Stride[0] * Stride[
            1] * 0.3:
            tile_name = img_name.split('.')[0] + '_' + str(X[i, j]) + '_' + str(Y[i, j]) + '_' + str(
                Stride[0]) + '_' + str(Stride[1]) + '_' + '.png'

            if color_norm == True:
                try:
                    Tile = normalizer_g.transform(Tile)
                except:
                    print('i=%d,j=%d' % (i, j))
                    continue

            img = Image.fromarray(Tile)
            img.save(dest_imagePath + tile_name)

            # for debug
            # if debug_g==True:
            #     pred_gg[i,j]=255

def parallel_tiling_roi(i,X,Y,dest_imagePath,img_name,Stride,File,color_norm,roi_mask):
    Slide = openslide.OpenSlide(File)

    for j in range(X.shape[1] - 1):
        Tile = Slide.read_region((int(X[i, j]), int(Y[i, j])), 0, (Stride[0], Stride[1]))
        Tile = np.asarray(Tile)
        Tile = Tile[:, :, :3]
        bn = np.sum(Tile[:, :, 0] < 5) + np.sum(np.mean(Tile,axis=2) > 245)
        if (np.std(Tile[:, :, 0]) + np.std(Tile[:, :, 1]) + np.std(Tile[:, :, 2])) / 3 > 18 and bn < Stride[0] * Stride[
            1] * 0.3 and roi_mask[i,j]==1:
            tile_name = img_name.split('.')[0] + '_' + str(X[i, j]) + '_' + str(Y[i, j]) + '_' + str(
                Stride[0]) + '_' + str(Stride[1]) + '_' + '.png'

            if color_norm == True:
                try:
                    Tile = normalizer_g.transform(Tile)
                except:
                    print('i=%d,j=%d' % (i, j))
                    continue

            img = Image.fromarray(Tile)
            img.save(dest_imagePath + tile_name)


def wsi_tiling(File,dest_imagePath,img_name,Tile_size,color_norm=False, tumor_mask=None, debug=False,parallel_running=True):
    since = time.time()
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
    if debug==True:
        pred_g = np.zeros((X.shape[0] - 1, X.shape[1] - 1, 3), 'uint8')
        global pred_gg
        pred_gg=pred_g

        global debug_g
        debug_g=debug

    if parallel_running==True and tumor_mask==None:
        # parallel-running
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
             for _ in executor.map(parallel_tiling, list(range(X.shape[0]-1)), repeat(X), repeat(Y), repeat(dest_imagePath),repeat(img_name),
                                   repeat(Stride),repeat(File),repeat(color_norm)):
                 pass
    elif parallel_running==True and tumor_mask!=None:
        tumor_mask=plt.imread(tumor_mask+img_name[:-5]+'.png')
        tumor_mask=transform.resize(tumor_mask,(X.shape[0]-1,X.shape[1]-1),order=0)
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
             for _ in executor.map(parallel_tiling_roi, list(range(X.shape[0]-1)), repeat(X), repeat(Y), repeat(dest_imagePath),repeat(img_name),
                                   repeat(Stride),repeat(File),repeat(color_norm),repeat(tumor_mask)):
                 pass
    else: # for debug
        for i in range(150,X.shape[0] - 1):
            for j in range(X.shape[1] - 1):
                    Tile = Slide.read_region((int(X[i, j]), int(Y[i, j])), 0, (Stride[0], Stride[1]))
                    Tile = np.asarray(Tile)
                    Tile = Tile[:, :, :3]
                    bn=np.sum(Tile[:, :, 0] < 5) + np.sum(np.mean(Tile,axis=2) > 245)
                    if (np.std(Tile[:,:,0])+np.std(Tile[:,:,1])+np.std(Tile[:,:,2]))/3>18 and bn<Stride[0]*Stride[1]*0.3:
                        tile_name=img_name.split('.')[0]+'_'+str(X[i,j])+'_'+str(Y[i,j])+'_'+str(Stride[0])+'_'+str(Stride[1])+'_'+'.png'
                        img = Image.fromarray(Tile)
                        img.save(dest_imagePath+tile_name)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

if __name__=='__main__':

    kang_colon_slide=False
    lee_gastric_slide=False
    tcga_coad_read_slide=False
    lee_colon_slide=True

    ## to tiling yonsei data
    if kang_colon_slide==True:
        imagePath=['../../data/kang_colon_slide/181119/',
                   '../../data/kang_colon_slide/181211/',
                   '../../data/kang_colon_slide/Kang_MSI_WSI_2019_10_07/']

        destPath=['../../data/pan_cancer_tils/data_yonsei_v01/181119_v2/',
                  '../../data/pan_cancer_tils/data_yonsei_v01/181211_v2/',
                  '../../data/pan_cancer_tils/data_yonsei_v01/Kang_MSI_WSI_2019_10_07_v2/']

        wsi_ext='.mrxs'
        # tileSize=[50,50] # micro-meters
        tileSize = [112, 112]  # micro-meters
    elif lee_gastric_slide == True:  ## to tiling lee data
        imagePath = ['../../data/lee_gastric_slide/Stomach_Immunotherapy/']
        destPath = ['../../data/pan_cancer_tils/data_lee_gastric/']
        wsi_ext='.tiff'
        # tileSize=[50,50] # micro-meters
        tileSize = [112, 112]  # micro-meters
    elif tcga_coad_read_slide==True:
        imagePath = ['../../data/tcga_coad_slide/tcga_coad/quality_a1/',
                     '../../data/tcga_coad_slide/tcga_coad/quality_a2/',
                     '../../data/tcga_coad_slide/tcga_coad/quality_b/',
                     '../../data/tcga_coad_slide/tcga_coad/quality_uncertain/',
                     '../../data/tcga_read_slide/dataset/']
        destPath = ['../../data/tcga_coad_read_data/coad_read_tissue_tiles/tcga_coad_a1/',
                    '../../data/tcga_coad_read_data/coad_read_tissue_tiles/tcga_coad_a2/',
                    '../../data/tcga_coad_read_data/coad_read_tissue_tiles/tcga_coad_b/',
                    '../../data/tcga_coad_read_data/coad_read_tissue_tiles/tcga_coad_uncertain/',
                    '../../data/tcga_coad_read_data/coad_read_tissue_tiles/tcga_read/']
        wsi_ext='.svs'
        # tileSize=[50,50] # micro-meters
        tileSize = [112, 112]  # micro-meters

    elif lee_colon_slide ==True:
        # switch=1-> tils detection tiling
        # swith=2-> tumor detection tiling
        # swith=3-> tumor region tiling -> e.g., msi prediction
        switch=3

        imagePath = ['../../data/Colon_St_Mary_Hospital_SungHak_Lee_Whole_Slide_Image/CRC St. Mary hospital/']

        wsi_ext='.tiff'

        clinic_info=pd.read_excel('../../data/lee_colon_data/Colorectal cancer dataset.xlsx')
        pid=[i+j for i, j in zip(clinic_info['S no (primary)'].tolist(),clinic_info['Sub no (T)'].tolist())]
        pid2 = [sub.replace('#', '-') for sub in pid if isinstance(sub, str)]

        if switch==1:
            tileSize = [112, 112]  # micro-meters
            destPath = ['../../data/lee_colon_data/all_tiles_tils/']
            color_norm=False
        elif switch==2:
            tileSize = [256, 256]  # micro-meters
            destPath = ['../../data/lee_colon_data/all_tiles_tumor/']
            color_norm = True
        elif switch==3:
            tileSize = [248.6272,248.6272] # note that: msi prediction model trained on this scale
            #tileSize = [256, 256]  # micro-meters
            destPath= ['../../data/lee_colon_data/msi_tiles_tumor_no_color_norm/']
            color_norm=False

            tumor_mask_path='../../data/lee_colon_data/tumor_pred/pred_masks/'
        else:
            raise RuntimeError('undefined selection .........')

    else:
        raise ValueError('incorrect data selection~~~~~~')

    if color_norm==True:
        reference_path=rela_path+'xhm_deep_learning/functions/macenko_reference_img.png'
        try:
            # Initialize the Macenko normalizer
            reference_img = np.array(
                Image.open(reference_path).convert('RGB'))
            normalizer = MacenkoNormalizer()
            normalizer.fit(reference_img)

            global normalizer_g
            normalizer_g = normalizer
        except:
            print('no given reference image for color normalization~~~~~')

    for i in range(len(imagePath)):
        temp_imagePath = imagePath[i]
        dest_imagePath = destPath[i]
        wsis = sorted(os.listdir(temp_imagePath))
        for img_name in tqdm(wsis):
            if wsi_ext in img_name:
                if lee_colon_slide==True:   # add this condition, only process tumor slides
                    temp_split=img_name.split('-')
                    temp_split[1]=temp_split[1].zfill(6)
                    pp='-'.join(temp_split)
                    if pp[:-5] in pid2:
                        file = temp_imagePath + img_name
                        print(img_name)
                        if switch==2:
                            wsi_tiling(file, dest_imagePath, img_name, tileSize, color_norm)
                        elif switch==3:
                            wsi_tiling(file, dest_imagePath, img_name, tileSize, color_norm,tumor_mask_path)
                        else:
                            raise RuntimeError('undefined options........')

                else:
                    file = temp_imagePath + img_name
                    wsi_tiling(file, dest_imagePath, img_name, tileSize)
