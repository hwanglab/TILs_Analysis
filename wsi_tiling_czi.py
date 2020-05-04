'''
divide .czi whole slide image into tiles and save the tiles into local disk

e.g., tils tiling 112x112um
      tumor detection tiling 256x256um

background tiles are removed based on some criteira (see code)

purpose: tiling the whole slide images
author: HONGMING XU, CCF
email: mxu@ualberta.ca
'''

import os
import sys
from tqdm import tqdm
import time
import czifile as czi
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import concurrent.futures
from itertools import repeat

rela_path='../../'
sys.path.insert(0,rela_path+'xhm_deep_learning/functions')
from MacenkoNormalizer import MacenkoNormalizer

def parallel_tiling(i,X,Y,dest_imagePath,img_name,Stride):

    for j in range(X.shape[1] - 1):
        Tile = c_wsi[int(X[i, j]):int(X[i, j] + Stride[0]), int(Y[i, j]):int(Y[i, j] + Stride[1]), :]
        bn = np.sum(Tile[:, :, 0] < 5) + np.sum(np.mean(Tile, axis=2) > 210)
        if (np.std(Tile[:, :, 0]) + np.std(Tile[:, :, 1]) + np.std(Tile[:, :, 2])) / 3 > 18 and bn < Stride[0] * Stride[
            1] * 0.3:
            tile_name = img_name.split('.')[0] + '_' + str(X[i, j]) + '_' + str(Y[i, j]) + '_' + str(
                Stride[0]) + '_' + str(Stride[1]) + '_' + '.png'

            if normalization == True:
                try:
                    Tile = normalizer_g.transform(Tile)
                except:
                    print('i=%d,j=%d' % (i, j))
                    continue

            img = Image.fromarray(Tile)
            img.save(dest_imagePath + tile_name)

def wsi_tiling_czi(File,dest_imagePath,img_name,Tile_size,debug=False,parallel_running=True,normalization=False):
    since = time.time()
    # open image

    czi_obj = czi.CziFile(File)

    #num_tiles = len(czi_obj.filtered_subblock_directory)
    #tile_dims_dict =czi_obj.metadata(raw=False)['ImageDocument']['Metadata']['Experiment']['ExperimentBlocks']['AcquisitionBlock'][
    #    'SubDimensionSetups']['RegionsSetup']['SampleHolder']['TileDimension']

    global c_wsi
    c_wsi = np.zeros(czi_obj.shape[2:], np.uint8)
    for i, directory_entry in enumerate(czi_obj.filtered_subblock_directory):
        subblock = directory_entry.data_segment()
        tile = subblock.data(resize=False, order=0)
        xs=directory_entry.start[2]-czi_obj.start[2]
        xe=xs+tile.shape[2]
        ys=directory_entry.start[3]-czi_obj.start[3]
        ye=ys+tile.shape[3]

        c_wsi[xs:xe,ys:ye,:]=tile.squeeze()


        # for i, directory_entry in enumerate(czi_obj.filtered_subblock_directory):
        #     subblock = directory_entry.data_segment()
        #     tile = subblock.data(resize=False, order=0)
        #     print('Size of tile:', str(tile.shape))
        #     index = tuple(slice(i - j, i - j + k) for i, j, k in zip(directory_entry.start, czi_obj.start, tile.shape))
        #     print('Location of tile: (channel, always-0, x-coordinates, y-coordinates, rgb)\n\t{}'.format(str(index)))
        #     #plt.imshow(tile.squeeze())
        #     #plt.title('Tile {} of {}'.format(i, num_tiles))
        #     #plt.show()

    # np_img = czi.imread(File,max_workers=10)
    # # overlay the channels
    # global np_img_mod
    # for i in range(np_img.shape[0]):
    #     if i == 0:
    #         np_img_mod = np_img[i, 0]
    #     else:
    #         np_img_mod = np.where(np_img_mod == 0,
    #                               np_img[i, 0],
    #                               np_img_mod)


    #xr = float(Slide.properties['openslide.mpp-x'])  # pixel resolution at x direction
    #yr = float(Slide.properties['openslide.mpp-y'])  # pixel resolution at y direction
    xr=czi_obj.metadata(raw=False)['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][0]['Value']*1e+6
    yr=czi_obj.metadata(raw=False)['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][1]['Value']*1e+6
    # generate X, Y coordinates for tiling
    Stride = [round(Tile_size[0] / xr), round(Tile_size[1] / yr)]
    Dims = c_wsi.shape
    X = np.arange(0, Dims[0] + 1, Stride[0])
    Y = np.arange(0, Dims[1] + 1, Stride[1])
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

    if parallel_running==True:
        # parallel-running
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
             for _ in executor.map(parallel_tiling, list(range(X.shape[0]-1)), repeat(X), repeat(Y), repeat(dest_imagePath),repeat(img_name),repeat(Stride)):
                 pass
    else: # for debug
        for i in range(X.shape[0] - 1):
            for j in range(X.shape[1] - 1):
                    Tile = c_wsi[int(X[i, j]):int(X[i,j]+Stride[0]),int(Y[i, j]):int(Y[i,j]+Stride[1]),:]
                    #Tile = np.asarray(Tile)
                    #Tile = Tile[:, :, :3]
                    bn=np.sum(Tile[:, :, 0] < 5) + np.sum(np.mean(Tile,axis=2) > 210)
                    if (np.std(Tile[:,:,0])+np.std(Tile[:,:,1])+np.std(Tile[:,:,2]))/3>18 and bn<Stride[0]*Stride[1]*0.3:
                        tile_name=img_name.split('.')[0]+'_'+str(X[i,j])+'_'+str(Y[i,j])+'_'+str(Stride[0])+'_'+str(Stride[1])+'_'+'.png'

                        if normalization == True:
                            try:
                                Tile = normalizer_g.transform(Tile)
                            except:
                                print('i=%d,j=%d' % (i, j))
                                continue

                        img = Image.fromarray(Tile)
                        img.save(dest_imagePath+tile_name)

    time_elapsed = time.time() - since
    print('Tiling complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

if __name__=='__main__':
    tils_tiling=False
    tumor_tiling=True

    rela_path='../../'
    imagePath = ['Z:/Datasets/Stomach_Cancer_Stage_4_Immunotherapy_Slide_image_Yonsei_JaeHo_Cheong/Stomach_Cancer_Stage4_Immunotherapy/biopsy_45pts/',
                 'Z:/Datasets/Stomach_Cancer_Stage_4_Immunotherapy_Slide_image_Yonsei_JaeHo_Cheong/Stomach_Cancer_Stage4_Immunotherapy/surgical_19pts/']

    if tils_tiling==True:
        destPath = [rela_path+'data/cheong_stomach_stage4/all_tiles_tils/biopsy/',
                    rela_path+'data/cheong_stomach_stage4/all_tiles_tils/surgery/']
        tileSize = [112, 112]  # micro-meters

        normalization=False
    elif tumor_tiling==True:
        destPath = [rela_path + 'data/cheong_stomach_stage4/all_tiles_tumor/biopsy/',
                    rela_path + 'data/cheong_stomach_stage4/all_tiles_tumor/surgery/']
        tileSize = [256, 256]  # micro-meters

        normalization=True
    else:
        raise RuntimeError('unexpected configrations.......')

    wsi_ext = '.czi'

    if normalization==True:
        reference_path='Y:/projects/xhm_deep_learning/functions/macenko_reference_img.png'
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
                file = temp_imagePath + img_name
                wsi_tiling_czi(file, dest_imagePath, img_name, tileSize,parallel_running=False,normalization=normalization)