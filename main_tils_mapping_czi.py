'''
purpose: generate tils detection maps
author: HONGMING XU, CCF, 2020
qeutions: mxu@ualberta.ca
'''
import os
import numpy as np
import czifile as czi
import scipy
import time
import matplotlib.pyplot as plt
from PIL import Image
import concurrent.futures
from itertools import repeat
import pandas as pd
import time
from tqdm import tqdm
import sys

rela_path='../../'

sys.path.insert(0,rela_path+'xhm_deep_learning/functions')
#from wsi_coarse_level import wsi_coarse_level

global t_g



def parallel_filling_czi(i,X,Y,img_name,Stride):

    for j in range(X.shape[1] - 1):
        Tile = c_wsi[int(X[i, j]):int(X[i, j] + Stride[0]), int(Y[i, j]):int(Y[i, j] + Stride[1]), :]
        bn = np.sum(Tile[:, :, 0] < 5) + np.sum(np.mean(Tile, axis=2) > 210)
        if (np.std(Tile[:, :, 0]) + np.std(Tile[:, :, 1]) + np.std(Tile[:, :, 2])) / 3 > 18 and bn < Stride[0] * Stride[
            1] * 0.3:
            tile_name = img_name.split('.')[0] + '_' + str(X[i, j]) + '_' + str(Y[i, j]) + '_' + str(
                Stride[0]) + '_' + str(Stride[1]) + '_' + '.png'
            ind = np.where(df_g['Name'] == tile_name)
            if len(ind[0]) > 0:  # ensure that ind is not empty, as some poor quality tiles maybe removed manually
                index = int(ind[0])
                pred_g_g[i, j, :] = list(df_g['Pred'])[index] * 255
                if df_g['Pred'][index] > t_g:
                    pred_c_g[i, j, 0] = 255
                else:
                    pred_c_g[i, j, 2] = 255

def wsi_tiling_czi(File,temp_predPath,dest_imagePath,img_name,Tile_size,parallel_running):
    since = time.time()

    df=pd.read_excel(temp_predPath+img_name.split('.')[0]+'.xlsx')
    # open image
    czi_obj = czi.CziFile(File)

    # num_tiles = len(czi_obj.filtered_subblock_directory)
    # tile_dims_dict =czi_obj.metadata(raw=False)['ImageDocument']['Metadata']['Experiment']['ExperimentBlocks']['AcquisitionBlock'][
    #    'SubDimensionSetups']['RegionsSetup']['SampleHolder']['TileDimension']

    global c_wsi
    c_wsi = np.zeros(czi_obj.shape[2:], np.uint8)
    for i, directory_entry in enumerate(czi_obj.filtered_subblock_directory):
        subblock = directory_entry.data_segment()
        tile = subblock.data(resize=False, order=0)
        xs = directory_entry.start[2] - czi_obj.start[2]
        xe = xs + tile.shape[2]
        ys = directory_entry.start[3] - czi_obj.start[3]
        ye = ys + tile.shape[3]

        c_wsi[xs:xe, ys:ye, :] = tile.squeeze()

    xr = czi_obj.metadata(raw=False)['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][0]['Value'] * 1e+6
    yr = czi_obj.metadata(raw=False)['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][1]['Value'] * 1e+6
    # generate X, Y coordinates for tiling
    Stride = [round(Tile_size[0] / xr), round(Tile_size[1] / yr)]
    Dims = c_wsi.shape
    X = np.arange(0, Dims[0] + 1, Stride[0])
    Y = np.arange(0, Dims[1] + 1, Stride[1])
    X, Y = np.meshgrid(X, Y)


    pred_c=np.zeros((X.shape[0]-1,X.shape[1]-1,3),'uint8')
    pred_g=np.zeros((X.shape[0]-1,X.shape[1]-1,3),'float')

    if parallel_running==True:
        global pred_c_g
        pred_c_g=pred_c
        global pred_g_g
        pred_g_g=pred_g
        global df_g
        df_g=df

        # parallel-running
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
             for _ in executor.map(parallel_filling_czi, list(range(X.shape[0]-1)), repeat(X), repeat(Y),repeat(img_name),repeat(Stride)):
                 pass

        img1 = Image.fromarray(pred_c_g)
        img2 = Image.fromarray(pred_g_g.astype('uint8'))
        img1.save(dest_imagePath + img_name.split('.')[0] + '_' + 'color.png')
        img2.save(dest_imagePath + img_name.split('.')[0] + '_' + 'gray.png')
    else: # sequential running for debugging
        for i in range(X.shape[0] - 1):
            for j in range(X.shape[1] - 1):
                Tile = c_wsi[int(X[i, j]):int(X[i, j] + Stride[0]), int(Y[i, j]):int(Y[i, j] + Stride[1]), :]
                bn = np.sum(Tile[:, :, 0] < 5) + np.sum(np.mean(Tile, axis=2) > 210)
                if (np.std(Tile[:, :, 0]) + np.std(Tile[:, :, 1]) + np.std(Tile[:, :, 2])) / 3 > 18 and bn < Stride[0] * \
                        Stride[
                            1] * 0.3:
                    tile_name = img_name.split('.')[0] + '_' + str(X[i, j]) + '_' + str(Y[i, j]) + '_' + str(
                        Stride[0]) + '_' + str(Stride[1]) + '_' + '.png'
                    ind = np.where(df_g['Name'] == tile_name)
                    if len(ind[0]) > 0:  # ensure that ind is not empty, as some poor quality tiles maybe removed manually
                        index = int(ind[0])
                        pred_g_g[i, j, :] = list(df_g['Pred'])[index] * 255
                        if df_g['Pred'][index] > t_g:
                            pred_c_g[i, j, 0] = 255
                        else:
                            pred_c_g[i, j, 2] = 255


    time_elapsed = time.time() - since
    print('Mapping complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

if __name__=='__main__':

    cheong_stomach=True

    if cheong_stomach==True:
        imagePath = [
            'Z:/Datasets/Stomach_Cancer_Stage_4_Immunotherapy_Slide_image_Yonsei_JaeHo_Cheong/Stomach_Cancer_Stage4_Immunotherapy/biopsy_45pts/',
            'Z:/Datasets/Stomach_Cancer_Stage_4_Immunotherapy_Slide_image_Yonsei_JaeHo_Cheong/Stomach_Cancer_Stage4_Immunotherapy/surgical_19pts/']
        predPath=[rela_path+'data/cheong_stomach_stage4/tils_pred/pred_excels/biopsy/',
                  rela_path+'data/cheong_stomach_stage4/tils_pred/pred_excels/surgery/']
        destPath=[rela_path+'data/cheong_stomach_stage4/tils_pred/pred_images/biopsy/',
                  rela_path+'data/cheong_stomach_stage4/tils_pred/pred_images/surgery/']
        wsi_ext='.czi'
        t_g=0.3
    else:
        raise RuntimeError('incorrect selection of dataset........')

    #tileSize=[50,50] # micro-meters
    tileSize=[112,112] # micro-meters
    parallel_running=False # True for parallel running
    for i in range(len(imagePath)):
        temp_imagePath = imagePath[i]
        temp_predPath = predPath[i]
        dest_imagePath = destPath[i]
        wsis = sorted(os.listdir(temp_imagePath))
        for img_name in wsis:
            if wsi_ext in img_name:
                file=temp_imagePath+img_name
                print(img_name)
                wsi_tiling_czi(file, temp_predPath, dest_imagePath,img_name, tileSize, parallel_running=parallel_running)