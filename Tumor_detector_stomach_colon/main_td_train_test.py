'''
main function to train and test tumor detection for gastric and colon cancer slides
    train: model is trained using tcga datasets released by Nature Medicine, Jacob 2019
    test: including internal testing and external testing
    main functions: see Transfer_Learning_PyTorch_V01.py class

author: Hongming Xu, CCF, 2020
questions: mxu@ualberta.ca
'''

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7" # use gpu 4
import sys
import pandas as pd
import time
import argparse

rela_path='../../../'
sys.path.insert(0,rela_path+'xhm_deep_learning/models')
sys.path.insert(0,rela_path+'xhm_deep_learning/functions')
from Transfer_Learning_PyTorch_V01 import Transfer_Learning_PyTorch_V01             # Transfer_Learning is my defined class


# switch between training & testing & testing_external
training=False
testing=False
testing_ext=True

if __name__=='__main__':

    if training == True:
        data_dir = rela_path + 'data/tcga_gas_col_data/data_v01/'

        model_dir = './models/'

        model_version = []
        validation_acc = []
        testing_acc = []
        training_time = []

        # parameter settings
        model_name = ['resnet18']
        frozen_per = [0, 0.8]  # percentile of frozen trainable layers, typically 0,0.5,0.8 [0,1]
        optimizer = ['sgd', 'adam']
        learning_rate = [0.001, 0.0001]
        batch_size = [4, 16, 64]

        load_data = 'v2'  # change this one according to different applications
        num_workers = 10
        epochs = 100
        imagenet_init = True  # False - weights are randomly initialized, tune_all_layers will be run
        num_early_stoping = 5
        zscore = False

        for i in range(len(frozen_per)):
            fp = frozen_per[i]
            for j in range(len(optimizer)):
                op = optimizer[j]
                for k in range(len(learning_rate)):
                    lr = learning_rate[k]
                    for b in range(len(batch_size)):
                        bs = batch_size[b]

                        start_time = time.time()
                        model_tl = Transfer_Learning_PyTorch_V01(load_data, data_dir, model_dir, model_name[0], bs,
                                                                 num_workers, epochs,
                                                                 imagenet_init, fp, op, lr, num_early_stoping, zscore)

                        valid_acc, _ = model_tl.train_model()
                        print("---{} minutes---".format((time.time() - start_time) / 60))
                        training_time.append((time.time() - start_time) / 60)

                        model_v = "{}_{}_{}_{}_{}.pt".format(model_name[0], fp, op, lr, bs)
                        model_version.append(model_v)
                        validation_acc.append(valid_acc.cpu().numpy().tolist())

                        model_tl = Transfer_Learning_PyTorch_V01(load_data=load_data, test_dir=data_dir,
                                                                 model_dir=model_dir,
                                                                 model_name=model_name[0],
                                                                 batch_size=bs, fp=fp, op=op, lr=lr)
                        test_acc = model_tl.test_model()
                        testing_acc.append(test_acc.cpu().numpy().tolist())

        data = {'Models': model_version, 'Valid Acc': validation_acc, 'Test Acc': testing_acc,
                            'Training Time': training_time}
        df = pd.DataFrame(data)
        pred_file = model_dir + 'logs.xlsx'
        df.to_excel(pred_file)

    if testing_ext==True:
        # best_resnet18='resnet18_0_adam_0.0001_64.pt'


        lee_colon = True

        if lee_colon == True:
            test_path = [rela_path+'data/lee_colon_data/all_tiles_tumor/']
            wsi_path = [rela_path+'data/lee_colon_data/wsi_tumor_files/']
            output_path = [rela_path+'data/lee_colon_data/tumor_pred/pred_excels/']
            wsi_ext = '.tiff'

        else:
            raise RuntimeError('processing dataset selection is not correct~~~~~~~~~')

        for i in range(len(test_path)):
            start_time = time.time()
            # best resnet18
            model_tl = Transfer_Learning_PyTorch_V01(test_dir=test_path[i],
                                                     model_dir='./models/',
                                                     model_name='resnet18',
                                                     batch_size=64, fp=0, op='adam',
                                                     lr=0.0001, num_workers=10, wsi_path=wsi_path[i], wsi_ext=wsi_ext,
                                                     output_path=output_path[i])
            model_tl.test_model_external_temp_tumor()
            print("---{} minutes---".format((time.time() - start_time) / 60))