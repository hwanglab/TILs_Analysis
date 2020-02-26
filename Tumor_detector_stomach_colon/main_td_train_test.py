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
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2" # use gpu 4
import sys
import pandas as pd
import time
import argparse

rela_path='../../../'
sys.path.insert(0,rela_path+'xhm_deep_learning/models')
sys.path.insert(0,rela_path+'xhm_deep_learning/functions')
from Transfer_Learning_PyTorch_V01 import Transfer_Learning_PyTorch_V01             # Transfer_Learning is my defined class


# switch between training & testing & testing_external
training=True
testing=False
# testing_ext=False

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