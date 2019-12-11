"""
purpose: perform color normalization on tils dataset
reference image: macenko_reference_img.png

Notes: if image is biger than 300x300, we only select the central region 300x300 pixels
"""
import os
import sys
import glob
sys.path.insert(0,'../../../xhm_deep_learning/functions')
import pandas as pd
from MacenkoNormalizer import MacenkoNormalizer
import numpy as np
from PIL import Image
import concurrent.futures
import scipy
import time
import matplotlib.pyplot as plt
from itertools import repeat

data_v02=['../../../data/pan_cancer_tils/data_v02/train/tils/',
          '../../../data/pan_cancer_tils/data_v02/train/others/',
          '../../../data/pan_cancer_tils/data_v02/valid/tils/',
          '../../../data/pan_cancer_tils/data_v02/valid/others/',
          '../../../data/pan_cancer_tils/data_v02/test/tils/',
          '../../../data/pan_cancer_tils/data_v02/test/others/']

data_v03=['../../../data/pan_cancer_tils/data_v03/train/tils/',
          '../../../data/pan_cancer_tils/data_v03/train/others/',
          '../../../data/pan_cancer_tils/data_v03/valid/tils/',
          '../../../data/pan_cancer_tils/data_v03/valid/others/',
          '../../../data/pan_cancer_tils/data_v03/test/tils/',
          '../../../data/pan_cancer_tils/data_v03/test/others/']



def color_normalization():
    # Initialize the Macenko normalizer
    reference_img = np.array(
        Image.open('../../../xhm_deep_learning/functions/macenko_reference_img.png').convert('RGB'))
    normalizer = MacenkoNormalizer()
    normalizer.fit(reference_img)

    for k in range(len(data_v02)):
        tiles=sorted(glob.glob(data_v02[k]+'*.png'))

        start_time = time.time()

        with concurrent.futures.ProcessPoolExecutor(max_workers=30) as executor:
            for _ in executor.map(parallel_norm,tiles,repeat(normalizer),repeat(data_v03[k])):
                pass

        # debug non-parallel version
        #parallel_norm(tiles[0], normalizer, data_v03[k])

        print("---{} minutes---".format((time.time() - start_time) / 60))
        #print(rr2)
    print('the program has been done')

def parallel_norm(img,normalizer,dest_path):
    try:
        tile=np.array(Image.open(img))
        if tile.shape[0]>300 or tile.shape[1]>300:
            crow=round(tile.shape[0]/2)
            ccol=round(tile.shape[1]/2)
            tile=tile[crow-150:crow+150,ccol-150:ccol+150,:]

        if tile.shape[2]>3:
            tile=tile[:,:,0:3]

        tile = normalizer.transform(tile)
        img_cn = Image.fromarray(tile)
        #img_cn.save(dest_path+img.split('\\')[-1]) # win10
        img_cn.save(dest_path + img.split('/')[-1])  # linux

    except:
        print('img name is=%s' % img)
        print('cannot color normalization, skip it')
        #tile = np.array(Image.open(img))
        #img_cn = Image.fromarray(tile)
        # img.save(debug_path+tt.split('\\')[-1]) # win10
        #img_cn.save(dest_path + img.split('/')[-1])  # linux


if __name__=='__main__':
    color_normalization()