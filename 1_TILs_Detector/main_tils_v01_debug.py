'''
main function to train and test tils detection using tcga dataset
by Hongming Xu
CCF, 2019
'''

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "4" # use gpu 4
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
import sys
import pandas as pd
import numpy as np
import time
import scipy
import argparse
from tensorflow.keras.models import model_from_json
#-- for training
sys.path.insert(0,'../../../xhm_deep_learning/models')
sys.path.insert(0,'../../../xhm_deep_learning/functions')
#sys.path.append('../../xhm_deep_learning/models')
#sys.path.append('/home/xuh3/projects/xhm_deep_learning/models') # linux absolute path
from Transfer_Learning_V02_debug import Transfer_Learning_V02_debug             # Transfer_Learning is my defined class
from wsi_tiling_prediction_v4 import wsi_tiling_prediction_v4

# switch between training & testing
training=1
testing=0
model_option='model_resnet18_v03'

if __name__=='__main__':

    if training==1:
        #parameter settings
        train_dir = 'E:/data/data_v03/train/'
        valid_dir = 'E:/data/data_v03/valid/'

        model_dir = '../../../data/pan_cancer_tils/models_debug/'
        tensorboard_dir = '../../../data/pan_cancer_tils/tensorboard/'

        model_name=['resnet18','shufflenet','densenet']
        frozen_per = [0,0.8]  # percentile of frozen trainable layers, typically 0,0.5,0.8 [0,1]
        optimizer=['sgd','adam']
        learning_rate=[0.001,0.0001,0.00001]

        batch_size = 64
        epochs = 100
        gpus=0 # 0 - run the model by defaul assignment of gpu
        imagenet_init=True # False - weights are randomly initialized, tune_all_layers will be run
        zscore=False         # note setting for this one???? it is related to testing process


        fp=frozen_per[0]

        op=optimizer[0]

        lr=learning_rate[1]
        start_time = time.time()
        resnet18_model=Transfer_Learning_V02_debug(train_dir,valid_dir,model_dir,tensorboard_dir,
                                                                model_name[0],batch_size,epochs,gpus,
                                                                imagenet_init,fp,op,lr,zscore)

        resnet18_model.train_model()
        print("---{} minutes---".format((time.time() - start_time) / 60))

    elif testing==1:
        if model_option=='model_resnet18_v01':
            # load model
            model_dir='../../../data/kang_colon_data/output_msi/tl_models/'
            json_file = open(model_dir + model_option+'.json', 'r')

            output_mask_path = '../../../data/kang_colon_data/predictions_msi/resnet18_v01/masks/'
            heatmap_path = '../../../data/kang_colon_data/predictions_msi/resnet18_v01/heatmaps/'

        elif model_option=='model_resnet18_v02':
            pass
        elif model_option=='model_resnet18_v03':
            # load model
            model_dir = '../../../data/kang_colon_data/output_msi/tl_models/'
            json_file = open(model_dir + model_option+'.json', 'r')

            output_mask_path = '../../../data/kang_colon_data/predictions_msi/resnet18_v03/masks/'
            heatmap_path = '../../../data/kang_colon_data/predictions_msi/resnet18_v03/heatmaps/'

        else:
            print('no input model~~~~~~')

        try:
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)

            # load weigths into new model
            loaded_model.load_weights(model_dir + model_option+'_weights.h5')
            print("loaded model from disk")
        except:
            print('load model errors~~~~~~~~')

        #parameter settings
        tile_size = 512
        magnification = 10
        data_path = './kang_colon_master_table.xlsx'
        df = pd.read_excel(data_path, sheet_name='Sheet1')

        imagePath=['../../../data/kang_colon_slide/181119/',
                   '../../../data/kang_colon_slide/181211/',
                   '../../../data/kang_colon_slide/Kang_MSI_WSI_2019_10_07/']

        tumor_mask_path=['../../../data/kang_colon_data/predictions_tumor/dl_model_v01/181119/',
                         '../../../data/kang_colon_data/predictions_tumor/dl_model_v01/181211/',
                         '../../../data/kang_colon_data/predictions_tumor/dl_model_v01/Kang_MSI_WSI_2019_10_07/']

        ref_path='../../../xhm_deep_learning/functions/macenko_reference_img.png'

        for i in range(len(imagePath)):
            temp_imagePath=imagePath[i]
            mask_path=tumor_mask_path[i]
            wsis=sorted(os.listdir(temp_imagePath))
            for img_name in wsis:
                if '.mrxs' in img_name:
                    if i==2:
                        pid=img_name[:-5]
                    else:
                        pid=temp_imagePath.split('/')[-2]+'_'+img_name[:-5]
                    ind1 = np.where((pid == df['patient ID']).to_numpy())[0]
                    if (len(ind1)==1 and df['data split'][ind1].values==3):
                        print(img_name)
                        start_time = time.time()
                        pmask,pred = wsi_tiling_prediction_v4(loaded_model, temp_imagePath+img_name, mask_path+img_name+'.png',
                                                         magnification, tile_size, heatmap_path, MappingMag=2.5, Coverage=0.5,
                                                         gt_Mask=False, Zscore=False, normalization=True, Scaling=True,reference_path=ref_path)
                        # print("---%f minutes---" % (time.time()-start_time)/60)
                        print("---{} minutes---".format((time.time() - start_time) / 60))
                        np.save(output_mask_path + pid + '.npy', pred)
                        #scipy.misc.imsave(output_mask_path + pid + '.png', pmask)

    else:
        print('no training or testing!!!!!!!!')