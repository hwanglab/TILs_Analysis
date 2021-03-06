'''
main function to train and test tils detection
    train: model is trained using pan-cancer tcga datasets
    test: including internal testing and external testing
    main functions: see Transfer_Learning_PyTorch_V01.py class

author: Hongming Xu, CCF, 2019
email: mxu@ualberta.ca

environment: pytorch
'''

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"                                    # The GPU id to use, usually either "0" or "1";
import sys
import pandas as pd
import time
import argparse

rela_path='../../'                                                                  # change the path adaptively
sys.path.insert(0,rela_path+'xhm_deep_learning/models')
sys.path.insert(0,rela_path+'xhm_deep_learning/functions')
#sys.path.append('/home/xuh3/projects/xhm_deep_learning/models')                    # linux absolute path
from Transfer_Learning_PyTorch import Transfer_Learning_PyTorch             # Transfer_Learning is my defined class


# switch between training & testing & testing_external
training=False                                                                      # used for model development by developer
testing_int=False                                                                   # for debugging, used by deveoper
testing_ext=False                                                                   # used by deveoper
testing_end_to_end=True                                                             # end_to_end testing: input-> the folder containing wsis
                                                                                    # easily used by users: output-> the prediction probability maps
if __name__=='__main__':

    if training == True:
        data_dir = rela_path+'data/pan_cancer_tils/data_v02/'                        # not color normalized version
        model_dir = rela_path+'data/pan_cancer_tils/models/resnet34/'
        model_version = []
        validation_acc = []
        testing_acc = []
        training_time = []

        # parameter settings
        model_name = ['resnet34', 'shufflenet', 'resnet18']
        frozen_per = [0, 0.8]                                                       # percentile of frozen trainable layers, typically 0,0.5,0.8 [0,1]
        optimizer = ['sgd', 'adam']
        learning_rate = [0.001, 0.0001, 0.00001]
        batch_size = [4, 16, 32, 64]

        load_data = 'v1'                                                            # change this one according to different applications
        num_workers=10
        epochs = 100
        imagenet_init=True                                                          # False - weights are randomly initialized, tune_all_layers will be run
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
                        model_tl=Transfer_Learning_PyTorch(load_data, data_dir, model_dir, model_name[0], bs, num_workers, epochs,
                                                                     imagenet_init,fp,op,lr,num_early_stoping,zscore)

                        valid_acc, _ = model_tl.train_model()
                        print("---{} minutes---".format((time.time() - start_time) / 60))
                        training_time.append((time.time() - start_time) / 60)

                        model_v="{}_{}_{}_{}_{}.pt".format(model_name[0], fp, op, lr, bs)
                        model_version.append(model_v)
                        validation_acc.append(valid_acc.cpu().numpy().tolist())


                        model_tl = Transfer_Learning_PyTorch(load_data=load_data,test_dir=data_dir, model_dir=model_dir,
                                                                 model_name=model_name[0],batch_size=bs,fp=fp,op=op,lr=lr)
                        test_acc = model_tl.test_model()
                        testing_acc.append(test_acc.cpu().numpy().tolist())

        data = {'Models': model_version, 'Valid Acc': validation_acc, 'Test Acc': testing_acc, 'Training Time': training_time}
        df = pd.DataFrame(data)
        pred_file = model_dir + 'logs.xlsx'
        df.to_excel(pred_file)

    if testing_int == True: # for debugging
        data_dir = rela_path+'data/pan_cancer_tils/data_v02/'  # not color normalized version
        model_dir = rela_path+'data/pan_cancer_tils/models/resnet18/'
       # parameter settings
        model_name = ['resnet18']
        frozen_per = [0]  # percentile of frozen trainable layers, typically 0,0.5,0.8 [0,1]
        optimizer = ['adam']
        learning_rate = [0.0001]
        batch_size = [4]

        load_data = 'v1'
        num_workers = 10
        epochs = 100
        imagenet_init = True
        num_early_stoping = 5
        zscore = False

        model_tl = Transfer_Learning_PyTorch(load_data=load_data, test_dir=data_dir,
                                                model_dir=model_dir,
                                                model_name=model_name[0],
                                                batch_size=batch_size[0], fp=frozen_per[0], op=optimizer[0], lr=learning_rate[0])
        test_acc = model_tl.test_model()

    if testing_ext==True:
        #best_resnet18='resnet18_0_adam_0.0001_4.pt'
        # switches to perform test on different datasets
        kang_colon=False
        lee_gastric=False
        tcga_coad_read=False
        lee_colon=False
        cheong_stomach=False

        if kang_colon==True:
            test_path=['../../../data/pan_cancer_tils/data_yonsei_v01/181119_v2/',
                       '../../../data/pan_cancer_tils/data_yonsei_v01/181211_v2/',
                       '../../../data/pan_cancer_tils/data_yonsei_v01/Kang_MSI_WSI_2019_10_07_v2/']

            wsi_path=['../../../data/kang_colon_slide/181119/',
                      '../../../data/kang_colon_slide/181211/',
                      '../../../data/kang_colon_slide/Kang_MSI_WSI_2019_10_07/']
            wsi_ext='.mrxs'
        elif lee_gastric==True:
            test_path=['../../../data/pan_cancer_tils/data_lee_gastric/']
            wsi_path=['../../../data/lee_gastric_slide/Stomach_Immunotherapy/']
            wsi_ext='.tiff'
        elif tcga_coad_read==True:
            test_path=[rela_path+'data/tcga_coad_read_data/coad_read_tissue_tiles/tcga_coad_a1/',
                       rela_path+'data/tcga_coad_read_data/coad_read_tissue_tiles/tcga_coad_a2/',
                       rela_path+'data/tcga_coad_read_data/coad_read_tissue_tiles/tcga_coad_b/',
                       rela_path+'data/tcga_coad_read_data/coad_read_tissue_tiles/tcga_coad_uncertain/',
                       rela_path+'data/tcga_coad_read_data/coad_read_tissue_tiles/tcga_read/']

            wsi_path=[rela_path+'data/tcga_coad_slide/tcga_coad/quality_a1/',
                      rela_path+'data/tcga_coad_slide/tcga_coad/quality_a2/',
                      rela_path+'data/tcga_coad_slide/tcga_coad/quality_b/',
                      rela_path+'data/tcga_coad_slide/tcga_coad/quality_uncertain/',
                      rela_path+'data/tcga_read_slide/dataset/']
            output_path=rela_path+'data/tcga_coad_read_data/coad_read_tils_preds/pred_files/'
            wsi_ext='.svs'
        elif lee_colon==True:
            test_path=[rela_path+'data/lee_colon_data/all_tiles_tils/']
            wsi_path=[rela_path+'data/lee_colon_data/wsi_tumor_files/']
            output_path=[rela_path+'data/lee_colon_data/tils_pred/pred_excels/']
            wsi_ext='.tiff'
        elif cheong_stomach==True:
            test_path=['../../data/cheong_stomach_stage4/all_tiles_tils/biopsy/',
                       '../../data/cheong_stomach_stage4/all_tiles_tils/surgery/']
            wsi_path=['../../data/Stomach_Cancer_Stage4_Immunotherapy/biopsy_45pts/',
                      '../../data/Stomach_Cancer_Stage4_Immunotherapy/surgical_19pts/']

            output_path=['../../data/cheong_stomach_stage4/tils_pred/pred_excels/biopsy/',
                         '../../data/cheong_stomach_stage4/tils_pred/pred_excels/surgery/']
            wsi_ext='.czi'

        else:
            raise RuntimeError('processing dataset selection is not correct~~~~~~~~~')

        for i in range(len(test_path)):
            start_time = time.time()
            # best resnet18
            model_tl = Transfer_Learning_PyTorch(test_dir=test_path[i], model_dir=rela_path+'data/pan_cancer_tils/models/resnet18/',
                                                       model_name='resnet18',
                                                       batch_size=4, fp=0, op='adam',
                                                       lr=0.0001,num_workers=10,wsi_path=wsi_path[i],wsi_ext=wsi_ext,output_path=output_path[i])
            model_tl.test_model_external()
            print("---{} minutes---".format((time.time() - start_time) / 60))

    if testing_end_to_end==True:
        #switches to select different datasets
        Stomach_Immunotherapy_stmary=False
        GC_SM2_stmary=False
        Stomach_Cancer_Stage4_Immunotherapy=False
        tcga_blca=False
        tcga_stad=False
        colon_IHC=False
        tcga_luad=False
        gastric_trial=False
        gastric_trial_czi=True

        cuda_id = 0
        class_name = ['others', 'tils']  # see data fold names
        class_interest = 1

        if Stomach_Immunotherapy_stmary==True:
            wsi_path=[rela_path+'data/stomach_cancer_immunotherapy/Stomach_Immunotherapy_stmary/']
            output_path=[rela_path+'data/stomach_cancer_immunotherapy/tils_maps/Stomach_Immunotherapy_stmary/']
            wsi_ext='.tiff'
        elif GC_SM2_stmary==True:
            wsi_path = [rela_path + 'data/stomach_cancer_immunotherapy/GC_SM2_stmary/']
            output_path = [rela_path + 'data/stomach_cancer_immunotherapy/tils_maps/GC_SM2_stmary/']
            wsi_ext = '.svs'
        elif Stomach_Cancer_Stage4_Immunotherapy==True:
            wsi_path = [rela_path + 'data/stomach_cancer_immunotherapy/Stomach_Cancer_Stage4_Immunotherapy/biopsy_45pts/',
                        rela_path + 'data/stomach_cancer_immunotherapy/Stomach_Cancer_Stage4_Immunotherapy/surgical_19pts/']

            output_path = [rela_path + 'data/stomach_cancer_immunotherapy/tils_maps/Stomach_Cancer_Stage4_Immunotherapy/biopsy_45pts/',
                           rela_path + 'data/stomach_cancer_immunotherapy/tils_maps/Stomach_Cancer_Stage4_Immunotherapy/surgical_19pts/']
            wsi_ext='.czi'
        elif tcga_blca==True:
            wsi_path = [rela_path + 'data/tcga_blca_slide/blca_wsi/',
                        rela_path + 'data/tcga_blca_slide/blca_wsi2/']
            output_path = [rela_path + 'data/tcga_blca_slide/til_maps/wsi/',
                           rela_path + 'data/tcga_blca_slide/til_maps/wsi2/']
            wsi_ext='.svs'
        elif tcga_stad==True:
            wsi_path = [rela_path + 'data/tcga_stad_slide/wsis/']
            output_path = [rela_path + 'data/tcga_stad_slide/til_maps/wsis/']

            wsi_ext='.svs'
        elif colon_IHC==True:
            wsi_path = [rela_path + 'data/kang_colon_slide/colon_IHC_JK/']
            output_path = [rela_path + 'data/kang_colon_slide/colon_IHC_JK/til_maps/']
            wsi_ext='HE.mrxs'
        elif tcga_luad==True:
            wsi_path = [rela_path + 'data/tcga_luad_slide/LUAD_wsi/']
            output_path = [rela_path + 'data/tcga_luad_slide/til_maps/']
            wsi_ext='.svs'
        elif gastric_trial==True:
            # images are stored at labshared server
            wsi_path = ['/mnt/isilon/data/w_QHS/hwangt-share/Datasets/CLASSIC_Stomach_Cancer_Image/LEICA/1-100/',
                        '/mnt/isilon/data/w_QHS/hwangt-share/Datasets/CLASSIC_Stomach_Cancer_Image/LEICA/101-200/',
                        '/mnt/isilon/data/w_QHS/hwangt-share/Datasets/CLASSIC_Stomach_Cancer_Image/LEICA/201-300/',
                        '/mnt/isilon/data/w_QHS/hwangt-share/Datasets/CLASSIC_Stomach_Cancer_Image/LEICA/301-400/',
                        '/mnt/isilon/data/w_QHS/hwangt-share/Datasets/CLASSIC_Stomach_Cancer_Image/LEICA/401-500/',
                        '/mnt/isilon/data/w_QHS/hwangt-share/Datasets/CLASSIC_Stomach_Cancer_Image/LEICA/501-600/',
                        '/mnt/isilon/data/w_QHS/hwangt-share/Datasets/CLASSIC_Stomach_Cancer_Image/LEICA/601-622/']
            output_path = [rela_path + 'data/CLASSIC_stomach_cancer_image/til_maps/1-100/',
                           rela_path + 'data/CLASSIC_stomach_cancer_image/til_maps/101-200/',
                           rela_path + 'data/CLASSIC_stomach_cancer_image/til_maps/201-300/',
                           rela_path + 'data/CLASSIC_stomach_cancer_image/til_maps/301-400/',
                           rela_path + 'data/CLASSIC_stomach_cancer_image/til_maps/401-500/',
                           rela_path + 'data/CLASSIC_stomach_cancer_image/til_maps/501-600/',
                           rela_path + 'data/CLASSIC_stomach_cancer_image/til_maps/601-622/']
            wsi_ext = '.svs'
        elif gastric_trial_czi==True:
            wsi_path = [#'/mnt/isilon/data/w_QHS/hwangt-share/Datasets/CLASSIC_Stomach_Cancer_Image/sample_1_175/',
                        #'/mnt/isilon/data/w_QHS/hwangt-share/Datasets/CLASSIC_Stomach_Cancer_Image/sample_176_375/',
                        '/mnt/isilon/data/w_QHS/hwangt-share/Datasets/CLASSIC_Stomach_Cancer_Image/sample_376_556/',
                        '/mnt/isilon/data/w_QHS/hwangt-share/Datasets/CLASSIC_Stomach_Cancer_Image/sample_557_622/']
            output_path = [#rela_path + 'data/CLASSIC_stomach_cancer_image/til_maps/sample_1_175/',
                           #rela_path + 'data/CLASSIC_stomach_cancer_image/til_maps/sample_176_375/',
                           rela_path + 'data/CLASSIC_stomach_cancer_image/til_maps/sample_376_556/',
                           rela_path + 'data/CLASSIC_stomach_cancer_image/til_maps/sample_557_622/']
            wsi_ext = '.czi'
        else:
            raise RuntimeError('processing dataset selection is not correct~~~~~~~~~')

        for i in range(0,len(wsi_path)):
            start_time = time.time()
            # best resnet18
            model_tl = Transfer_Learning_PyTorch(model_dir=rela_path+'data/pan_cancer_tils/models/resnet18/',
                                                 model_name='resnet18', batch_size=4, fp=0, op='adam',
                                                 lr=0.0001,num_workers=20, wsi_path=wsi_path[i], wsi_ext=wsi_ext, output_path=output_path[i],
                                                 cuda_id=cuda_id, class_num=len(class_name),class_interest=class_interest,tile_size=[112,112])
            model_tl.test_end_to_end()
            print("---{} minutes---".format((time.time() - start_time) / 60))

