'''
main function to train and test tils detection using tcga dataset
by Hongming Xu
CCF, 2019
'''

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2" # use gpu 4
import sys
import pandas as pd
import numpy as np
import time
import scipy
import argparse
#-- for training
sys.path.insert(0,'../../../xhm_deep_learning/models')
sys.path.insert(0,'../../../xhm_deep_learning/functions')
#sys.path.append('../../xhm_deep_learning/models')
#sys.path.append('/home/xuh3/projects/xhm_deep_learning/models') # linux absolute path
from Transfer_Learning_PyTorch_V01_ext import Transfer_Learning_PyTorch_V01_ext             # Transfer_Learning is my defined class


# switch between training & testing
training=False
testing=False
testing_ext=True

if __name__=='__main__':

    data_dir = '../../../data/pan_cancer_tils/data_v02/'  # not color normalized version
    #data_dir='../../../tcga_gas_col/data_v01/'
    #model_dir = '../../../data/pan_cancer_tils/models_v02/'
    model_dir = '../../../data/pan_cancer_tils/models_v03/'
    model_version=[]
    validation_acc=[]
    testing_acc=[]
    training_time=[]

    #parameter settings
    model_name=['shufflenet','resnet18']
    frozen_per = [0,0.8]  # percentile of frozen trainable layers, typically 0,0.5,0.8 [0,1]
    optimizer=['sgd','adam']
    learning_rate=[0.001,0.0001,0.00001]
    batch_size = [4,16,32,64]

    if training == True:
        num_workers=10
        epochs = 100
        imagenet_init=True # False - weights are randomly initialized, tune_all_layers will be run
        num_early_stoping=5
        zscore=False

        for i in range(len(frozen_per)):
            fp=frozen_per[i]
            for j in range(len(optimizer)):
                op=optimizer[j]
                for k in range(len(learning_rate)):
                    lr=learning_rate[k]
                    for b in range(len(batch_size)):
                        bs=batch_size[b]

                        start_time = time.time()
                        model_tl=Transfer_Learning_PyTorch_V01_ext(data_dir, model_dir, model_name[0], bs, num_workers, epochs,
                                                                     imagenet_init,fp,op,lr,num_early_stoping,zscore)

                        valid_acc, _ = model_tl.train_model()
                        print("---{} minutes---".format((time.time() - start_time) / 60))
                        training_time.append((time.time() - start_time) / 60)

                        model_v="{}_{}_{}_{}_{}.pt".format(model_name[0], fp, op, lr, bs)
                        model_version.append(model_v)
                        validation_acc.append(valid_acc.cpu().numpy().tolist())


                        # testing
                        model_tl = Transfer_Learning_PyTorch_V01_ext(test_dir=data_dir, model_dir=model_dir,
                                                                       model_name=model_name[0],
                                                                       batch_size=bs,fp=fp,op=op,lr=lr)
                        test_acc = model_tl.test_model()
                        testing_acc.append(test_acc.cpu().numpy().tolist())




        data = {'Models': model_version, 'Valid Acc': validation_acc, 'Test Acc': testing_acc, 'Training Time': training_time}
        df = pd.DataFrame(data)
        pred_file = model_dir + 'logs.xlsx'
        df.to_excel(pred_file)

    if testing_ext==True:
        #best_resnet18='resnet18_0_adam_0.0001_4.pt'
        test_path='../../../data/pan_cancer_tils/data_yonsei_v01/test/181119/'
        #test_path='../../../data/pan_cancer_tils/data_yonsei_v01/'
        wsi_path='../../../data/kang_colon_slide/181119/'
        start_time = time.time()
        # best resnet18
        model_tl = Transfer_Learning_PyTorch_V01_ext(test_dir=test_path, model_dir='../../../data/pan_cancer_tils/models_v02/',
                                                       model_name='resnet18',
                                                       batch_size=4, fp=0, op='adam',
                                                       lr=0.0001,num_workers=10,wsi_path=wsi_path)
        model_tl.test_model_external()
        print("---{} minutes---".format((time.time() - start_time) / 60))

