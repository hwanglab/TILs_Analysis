'''
transfer learning class using pytorch
assume: different classes are saved into a local disk of different folders, e.g.,
train
     msi-h
     mss
valid
     msi-h
     mss
our class include transfer learning on different pre-trained models
     resent18
     resent34
     shufflenet

author: Hongming Xu, CCF, 2019
for comments and suggestions, send email to:
mxu@ualberta.ca
'''

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np

import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torchsummary import summary
import glob
from PIL import Image
import concurrent.futures
from itertools import repeat

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_curve,roc_auc_score

# local imports - functions created by Hongming Xu
from load_data import *
from wsi_tiling_pred import *
# local imports - meta-resnet functions downloaded online
from preact_resnet_meta import *

# class reference:https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        #original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        original_tuple = super().__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

class TestDataset(torch.utils.data.Dataset):
    def __init__(self,path,img_name,transform=None):
        self.img_paths=glob.glob(path+img_name+'*.png')
        #self.img_paths=os.listdir(path+img_name+'*.png')
        self.transform=transform

    def __getitem__(self,index):
        x = Image.open(self.img_paths[index])

        if self.transform is not None:
            x=self.transform(x)
        return x,self.img_paths[index]

    def __len__(self):
        return len(self.img_paths)

# class ValidCleanUnbiased(torch.utils.data.Dataset):
#     '''
#     load valid clear unbiased dataset for learning to reweight examples stuyding
#     '''
class Transfer_Learning_PyTorch:
    """
        A transfer learning object
        """

    def __init__(self, load_data=None, data_dir=None, model_dir=None, model_name=None, batch_size=64, num_workers=1,
                 epochs=100, img_init=True, fp=0.8, op='sgd', lr=0.0001, num_es=3,unbalanced=False,
                 zscore=False, test_dir=None, wsi_path=None,wsi_ext=None,output_path=None,
                 cuda_id=None,class_num=None,class_interest=None, tile_size=None, pred_parallel=False):
        self.load_data = load_data
        self.data_dir = data_dir
        self.model_dir = model_dir

        self.model_name = model_name
        self.batch_size = batch_size
        self.num_epochs = epochs
        self.num_workers = num_workers
        self.img_init = img_init
        self.fp = fp
        self.op = op
        self.lr = lr
        self.num_es = num_es
        self.unbalanced=unbalanced
        self.zscore = zscore

        self.test_dir = test_dir
        self.wsi_path = wsi_path
        self.wsi_ext = wsi_ext           # consider different wsi extenstion format, e.g., .mrxs, .tiff, .svs
        self.output_path = output_path
        self.cuda_id=cuda_id             # the gpu id
        self.class_num=class_num         # the number of class
        self.class_interest=class_interest # the index of interested output probability class
        self.tile_size=tile_size           # physical tile size - for end-to-end testing
        self.pred_parallel=pred_parallel

    @staticmethod
    def freeze_weights(module):
        for param in module.parameters():
            param.requires_grad = False

    @staticmethod
    def to_var2(x, device, requires_grad=True):
        # if torch.cuda.is_available():
        #    x = x.cuda()
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        return Variable(x, requires_grad=requires_grad)

    def recursively_enumerate_model(self, module):
        if list(module.children()) == []:
            return [module]
        else:
            enumerated_model = []
            for child in module.children():
                enumerated_model += self.recursively_enumerate_model(child)
            return enumerated_model

    def build_model(self, n_classes):
        if self.model_name == 'resnet18':
            model_ft = models.resnet18(pretrained=self.img_init)
        elif self.model_name =='shufflenet':
            model_ft = models.shufflenet_v2_x1_0(pretrained=self.img_init)
        elif self.model_name == 'resnet34':
            model_ft = models.resnet34(pretrained=self.img_init)
        else:
            raise ValueError('unrecognized model version: {}'.format(self.model_name))

        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, n_classes)


        layers = self.recursively_enumerate_model(model_ft)
        layers = [layer for layer in layers if (type(layer) != nn.BatchNorm2d and len(list(layer.parameters())) > 0)]
        for layer in layers[:round(self.fp * len(layers))]:
            self.freeze_weights(layer)

        #    raise ValueError('incorrect mode setting: {}'.format(mode))

        return model_ft
    def load_model(self):
        device = torch.device("cuda:" + str(self.cuda_id) if torch.cuda.is_available() else "cpu")
        #torch.cuda.set_device(device)
        model = self.build_model(self.class_num)
        model.to(device)

        model.load_state_dict(torch.load(self.model_dir + "{}_{}_{}_{}_{}.pt".format(
                                          self.model_name, self.fp, self.op, self.lr, self.batch_size)))
        #model.to(device)
        #model.cuda()
        model = nn.Sequential(model, nn.Softmax(dim=1))
        return model

    def train_model(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        
        dataloaders, dataset_sizes, class_names, count = load_data(self.data_dir, self.batch_size, self.num_workers,
                                                                                        self.load_data,mode='training')
        

        # build model
        model = self.build_model(len(class_names))
        # print(summary(model, input_size=(3, 224, 224)))


        # if torch.cuda.device_count()>1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     model = nn.DataParallel(model)

        model = model.to(device)
        #print(summary(model, input_size=(3, 224, 224)))

        # training setting, loss function, optimizer
        if self.unbalanced==True:
            ww = np.sum(count['train'][1]) / np.asarray(count['train'][1])
            class_weights = np.asarray(ww) / np.sum(ww)
            criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights)).cuda()
        else:
            criterion = nn.CrossEntropyLoss()
        if self.op == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        elif self.op == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
        else:
            raise ValueError('unrecognized optimization method: {}'.format(self.op))

        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

        # training process
        epochs_no_improve = 0
        valid_loss_min = np.Inf

        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        # ensure the folder name must be correct
        if not (list(dataloaders.keys()) == ['train', 'valid'] or list(dataloaders.keys()) == ['valid', 'train']):
            raise RuntimeError('data folder name must be train, and valid!!!!!!')

        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                running_sens = np.zeros(len(count[phase][0]),dtype=float)

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    for k in range(len(count[phase][0])):
                        running_sens[k]+=torch.sum(np.logical_and(preds.cpu() == labels.data.cpu(), labels.data.cpu() == count[phase][0][k]))

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                epoch_sen = 1-np.mean(running_sens/count[phase][1])

                print('{} Loss: {:.4f} Acc: {:.4f} Loss2: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc, epoch_sen))

                if phase == 'valid':
                    if self.unbalanced==True:  # for unbalanced data, we try to use epoch_sen for early stopping
                        epoch_loss=epoch_sen

                    if epoch_loss < valid_loss_min:  # using loss to early stop the model training
                        # save model
                        torch.save(model.state_dict(), self.model_dir +
                                   "{}_{}_{}_{}_{}.pt".format(self.model_name, self.fp, self.op, self.lr,
                                                              self.batch_size))
                        # track improvement
                        epochs_no_improve = 0
                        valid_loss_min = epoch_loss
                        best_acc = epoch_acc
                        best_epoch = epoch
                        best_model_wts = copy.deepcopy(model.state_dict())
                        # otherwise increment count if epochs with no improvement
                    else:
                        epochs_no_improve += 1
                        # triger early stopping
                        if epochs_no_improve >= self.num_es:
                            #print(f'\n Early Stopping!!!') # python3.7
                            print('\n Early Stopping!!!')   # python3.5

                            time_elapsed = time.time() - since
                            print('Training complete in {:.0f}m {:.0f}s'.format(
                                time_elapsed // 60, time_elapsed % 60))
                            print('Best val Acc: {:4f}'.format(best_acc))
                            # load best model weights
                            model.load_state_dict(best_model_wts)
                            return best_acc, model


    def train_model_lre(self):
        '''
        train the model using learning to reweight examples for robust deep learning
        '''
        #smoothing_alpha=0.9
        torch.cuda.set_device(1) # force to use the cuda1, otherwise always use cuda 0 by default
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


        dataloaders, dataset_sizes, class_names = load_data(self.data_dir, self.batch_size, self.num_workers,
                                                                   self.load_data,mode='training')
        ## ---load clean and unbiased data
        dataloaders_val,class_names2=load_data_valid_clean_unbiased(self.data_dir,self.num_workers)
        inputs_val=[]
        labels_val=[]
        for inputs,labels in dataloaders_val:
            inputs_val.append(inputs)
            labels_val.append(labels)
        inputs_val = torch.cat(inputs_val, dim=0)
        labels_val = torch.cat(labels_val, dim=0)
        inputs_val=self.to_var2(inputs_val,device,requires_grad=False)
        labels_val=self.to_var2(labels_val,device,requires_grad=False)

        # build model
        #model = TransferNet(self.model_name,len(class_names),self.img_init,self.fp)
        model = preact_resnet_meta18(num_classes=len(class_names))
        model = model.to(device)  # send model to a specific device
        torch.backends.cudnn.benchmark = True

        # print(summary(model, input_size=(3, 224, 224)))

        # training setting, loss function, optimizer
        criterion = nn.CrossEntropyLoss()
        criterion2 = nn.CrossEntropyLoss(reduction='none')
        if self.op == 'sgd':
            optimizer = optim.SGD(model.params(), lr=self.lr, momentum=0.9)
        elif self.op == 'adam':
            optimizer = optim.Adam(model.params(), lr=self.lr)
        else:
            raise ValueError('unrecognized optimization method: {}'.format(self.op))
        # Decay LR by a factor of 0.1 every 7 epochs
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

        # training process
        epochs_no_improve = 0
        valid_loss_min = np.Inf

        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        # ensure the folder name must be correct
        if not (list(dataloaders.keys()) == ['train', 'valid'] or list(dataloaders.keys()) == ['valid', 'train']):
            raise RuntimeError('data folder name must be train, and valid!!!!!!')

        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)

            model.train()  # Set model to training mode
            running_loss_train = 0.0
            running_corrects_train = 0
            # Iterate over data.
            for inputs, labels in dataloaders['train']:
                #inputs = inputs.to(device)
                #labels = labels.to(device)

                #meta_model=TransferNet(self.model_name,len(class_names),self.img_init,self.fp)
                meta_model=preact_resnet_meta18(num_classes=len(class_names))
                meta_model.load_state_dict(model.state_dict())

                meta_model.to(device)

                inputs=self.to_var2(inputs,device,requires_grad=False)
                labels=self.to_var2(labels,device,requires_grad=False)

                # Lines 4 - 5 initial forward pass to compute the initial weighted loss
                y_f_hat = meta_model(inputs)
                cost = criterion2(y_f_hat, labels)
                eps = self.to_var2(torch.zeros(cost.size()),device)
                l_f_meta = torch.sum(cost * eps)

                meta_model.zero_grad()

                # Line 6 perform a parameter update
                grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
                meta_model.update_params(self.lr, source_params=grads)

                # Line 8 - 10 2nd forward pass and getting the gradients with respect to epsilon
                y_g_hat = meta_model(inputs_val)

                l_g_meta = criterion(y_g_hat, labels_val)

                grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True)[0]

                # Line 11 computing and normalizing the weights
                w_tilde = torch.clamp(-grad_eps, min=0)
                norm_c = torch.sum(w_tilde)

                if norm_c != 0:
                    w = w_tilde / norm_c
                else:
                    w = w_tilde

                # Lines 12 - 14 computing for the loss with the computed weights
                # and then perform a parameter update
                y_f_hat = model(inputs)
                cost = criterion2(y_f_hat, labels)
                l_f = torch.sum(cost * w)

                optimizer.zero_grad()
                l_f.backward()
                optimizer.step()

                # statistics
                _, preds = torch.max(y_f_hat, 1)
                running_loss_train += l_f.item() * inputs.size(0)
                running_corrects_train += torch.sum(preds == labels.data)

                #meta_l = smoothing_alpha * meta_l + (1 - smoothing_alpha) * l_g_meta.item()
                #meta_losses_clean.append(meta_l / (1 - smoothing_alpha ** (i + 1)))

                #net_l = smoothing_alpha * net_l + (1 - smoothing_alpha) * l_f.item()
                #net_losses.append(net_l / (1 - smoothing_alpha ** (i + 1)))

            epoch_loss_train = running_loss_train / dataset_sizes['train']
            epoch_acc_train = running_corrects_train.double() / dataset_sizes['train']

            # evaluate valiation performance
            model.eval()  # Set model to evaluate mode
            running_loss_valid = 0.0
            running_corrects_valid = 0
            for inputs, labels in dataloaders['valid']:
                inputs = self.to_var2(inputs, device, requires_grad=False)
                labels = self.to_var2(labels, device, requires_grad=False)

                with torch.no_grad():
                #with torch.set_grad_enabled(False):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                # statistics
                running_loss_valid += loss.item() * inputs.size(0)
                running_corrects_valid += torch.sum(preds == labels.data)

            epoch_loss_valid = running_loss_valid / dataset_sizes['valid']
            epoch_acc_valid = running_corrects_valid.double() / dataset_sizes['valid']

            print('Train Loss: {:.4f} Train Acc: {:.4f} Valid Loss: {:.4f} Valid Acc: {:.4f}'.format(
                epoch_loss_train, epoch_acc_train,epoch_loss_valid, epoch_acc_valid))

            if epoch_loss_valid < valid_loss_min:  # using loss to early stop the model training
                # save model
                torch.save(model.state_dict(), self.model_dir +
                           "{}_{}_{}_{}_{}.pt".format(self.model_name, self.fp, self.op, self.lr,
                                                      self.batch_size))
                # track improvement
                epochs_no_improve = 0
                valid_loss_min = epoch_loss_valid
                best_acc = epoch_acc_valid
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                # otherwise increment count if epochs with no improvement
            else:
                epochs_no_improve += 1
                # triger early stopping
                if epochs_no_improve >= self.num_es:
                    # print(f'\n Early Stopping!!!') # python3.7
                    print('\n Early Stopping!!!')  # python3.5

                    time_elapsed = time.time() - since
                    print('Training complete in {:.0f}m {:.0f}s'.format(
                        time_elapsed // 60, time_elapsed % 60))
                    print('Best val Acc: {:4f}'.format(best_acc))
                    # load best model weights
                    model.load_state_dict(best_model_wts)
                    return best_acc, model



    def test_model(self):
        '''
        can be used for internal testing, the images are saved as the following structure:
        test
            class1
            class2
        output the prediction probabilities into excel file
            one column: imge name
            second column: prediction probabilities
        '''


        dataloaders, dataset_sizes, class_names = load_data(self.test_dir, self.batch_size, self.num_workers,
                                                                   self.load_data,mode='testing')

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # build model
        model = self.build_model(len(class_names))
        model.to(device)
        model.load_state_dict(torch.load(self.model_dir +
                                    "{}_{}_{}_{}_{}.pt".format(self.model_name, self.fp, self.op, self.lr,
                                                            self.batch_size)))
        model = nn.Sequential(model, nn.Softmax(dim=1))
        model.eval()

        running_corrects = 0
        imgs = []
        pred_tils = []
        for inputs, labels, paths in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)

            # imgs_n=[paths[i].split('\\')[-1] for i in range(len(paths))] #win10
            imgs_n = [paths[i].split('/')[-1] for i in range(len(paths))]  # win10
            preds_n = outputs.cpu().numpy()
            # preds_n=preds.cpu().numpy()
            # labels_n=labels.data.cpu().numpy()
            # pred_label=(preds_n==labels_n)
            imgs.extend(imgs_n)
            pred_tils.extend(preds_n[:, 1])

        data = {'Name': imgs, 'Pred': pred_tils}
        df = pd.DataFrame(data)
        pred_file = self.model_dir + "{}_{}_{}_{}_{}.pt".format(self.model_name, self.fp, self.op, self.lr,
                                                                self.batch_size) + '.xlsx'
        df.to_excel(pred_file)

        acc = running_corrects.double() / dataset_sizes['test']
        print('acc=%f' % acc)
        return acc

    def test_model_lre(self):
        '''
        can be used for internal testing, the images are saved as the following structure:
        test
            class1
            class2
        output the prediction probabilities into excel file
            one column: imge name
            second column: prediction probabilities
        '''

        dataloaders, dataset_sizes, class_names = load_data(self.test_dir, self.batch_size, self.num_workers,
                                                            self.load_data, mode='testing')

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # build model
        model = preact_resnet_meta18(num_classes=len(class_names))
        model = model.to(device)  # send model to a specific device
        #torch.backends.cudnn.benchmark = True
        model.load_state_dict(torch.load(self.model_dir +
                                         "{}_{}_{}_{}_{}.pt".format(self.model_name, self.fp, self.op, self.lr,
                                                                    self.batch_size)))
        model = nn.Sequential(model, nn.Softmax(dim=1))
        model.eval()

        running_corrects = 0
        imgs = []
        pred_tils = []
        for inputs, labels, paths in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)

            # imgs_n=[paths[i].split('\\')[-1] for i in range(len(paths))] #win10
            imgs_n = [paths[i].split('/')[-1] for i in range(len(paths))]  # win10
            preds_n = outputs.cpu().numpy()
            # preds_n=preds.cpu().numpy()
            # labels_n=labels.data.cpu().numpy()
            # pred_label=(preds_n==labels_n)
            imgs.extend(imgs_n)
            pred_tils.extend(preds_n[:, 1])

        data = {'Name': imgs, 'Pred': pred_tils}
        df = pd.DataFrame(data)
        pred_file = self.model_dir + "{}_{}_{}_{}_{}.pt".format(self.model_name, self.fp, self.op, self.lr,
                                                                self.batch_size) + '.xlsx'
        df.to_excel(pred_file)

        acc = running_corrects.double() / dataset_sizes['test']
        print('acc=%f' % acc)
        return acc

    def parallel_prediction(self,img_name,data_transforms,device,model,class_ind):

        if self.wsi_ext in img_name:
            pid=img_name.split('.')[0]
            print("patient id is %s" % pid)

            start_time = time.time()

            image_datasets = TestDataset(self.test_dir, pid, data_transforms)
            dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=self.batch_size,
                                                  shuffle=False, num_workers=self.num_workers)

            imgs = []
            preds = []
            for inputs, paths in dataloaders:
                inputs = inputs.to(device)

                with torch.no_grad():
                    outputs = model(inputs)

                # imgs_n=[paths[i].split('\\')[-1] for i in range(len(paths))] #win10
                imgs_n = [paths[i].split('/')[-1] for i in range(len(paths))]  # linux
                preds_n = outputs.cpu().numpy()

                imgs.extend(imgs_n)
                preds.extend(preds_n[:, class_ind])

            data = {'Name': imgs, 'Pred': preds}
            df = pd.DataFrame(data)

            df.to_excel(self.output_path + pid + '.xlsx')

            print((time.time() - start_time) / 60)

    def test_model_external(self):
        '''
        can be used for external testing, all the images (unknow labels) are put under the same folder

        each wsi has many tiles

        output predictions and save results for each patient into a separate excel file
        '''
        data_transforms=transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),])

        # build model
        #model = self.build_model(self.class_num)
        #model.to(device)
        #model.load_state_dict(torch.load(self.model_dir +"{}_{}_{}_{}_{}.pt".format(self.model_name, self.fp, self.op, self.lr,
        #                                                            self.batch_size)))
        #model = nn.Sequential(model, nn.Softmax(dim=1))
        device = torch.device("cuda:" + str(self.cuda_id) if torch.cuda.is_available() else "cpu")
        model = self.load_model()
        model.eval()

        wsis = sorted(os.listdir(self.wsi_path))
        class_ind=self.class_interest

        if self.pred_parallel==True:
            with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
                for _ in executor.map(self.parallel_prediction, wsis, repeat(data_transforms), repeat(device),
                                      repeat(model),repeat(class_ind)):
                    pass
        else:
            for img_name in wsis:
                if self.wsi_ext in img_name:
                    pid=img_name.split('.')[0]
                    print("patient id is %s" % pid)

                    start_time = time.time()

                    image_datasets = TestDataset(self.test_dir, pid, data_transforms)
                    # x,p=image_datasets.__getitem__(0)

                    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=self.batch_size,
                                                          shuffle=False, num_workers=self.num_workers)
                    if len(dataloaders)>0:
                        imgs = []
                        preds = []
                        for inputs, paths in dataloaders:
                            inputs = inputs.to(device)

                            with torch.no_grad():
                                outputs = model(inputs)

                                # imgs_n=[paths[i].split('\\')[-1] for i in range(len(paths))] #win10
                            imgs_n = [paths[i].split('/')[-1] for i in range(len(paths))]  # linux
                            preds_n = outputs.cpu().numpy()

                            imgs.extend(imgs_n)
                            preds.extend(preds_n[:, class_ind])

                        data = {'Name': imgs, 'Pred': preds}
                        df = pd.DataFrame(data)
                        df.to_excel(self.output_path + pid + '.xlsx')

                        print((time.time() - start_time) / 60)

        print('automatic prediction done!!!!')


    def test_model_internal_patient(self):
        '''
        internal testing, output patient-level prediction results

        images are saved in the following structure:
        test
            class1
            class2
        '''
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # read master table
        df_m=pd.read_excel('./kang_colon_master_table.xlsx')

        data_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(), ])

        # build model
        model = self.build_model(2)
        model.to(device)
        model.load_state_dict(torch.load(self.model_dir +
                                         "{}_{}_{}_{}_{}.pt".format(self.model_name, self.fp, self.op, self.lr,
                                                                    self.batch_size)))
        model = nn.Sequential(model, nn.Softmax(dim=1))
        model.eval()

        indicator = df_m['data split'].tolist()
        pid = df_m['patient ID'].tolist()
        gt_label = df_m['GT class'].tolist()

        pids = []
        preds = []
        gts = []
        for i,val in enumerate(indicator):
            if val==3:
                temp_p=pid[i]
                temp_gt=gt_label[i]
                print("patient id is %s" % temp_p)
                pids.append(temp_p)
                gts.append(temp_gt)

                if temp_gt=='MSI-H':
                    test_dir_path=self.test_dir+'msi_h/'
                elif temp_gt=='MSS':
                    test_dir_path=self.test_dir+'mss/'
                else:
                    raise ValueError('undefined gt class %s\n' % temp_gt )

                image_datasets = TestDataset(test_dir_path, temp_p, data_transforms)
                # x,p=image_datasets.__getitem__(0)

                dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=self.batch_size,
                                                          shuffle=False, num_workers=self.num_workers)


                pred_msi = []
                for inputs, paths in dataloaders:
                    inputs = inputs.to(device)

                    with torch.no_grad():
                        outputs = model(inputs)

                    preds_n = outputs.cpu().numpy()

                    pred_msi.extend(preds_n[:, 0])  # msi-h class corresponds to the first element

                preds.append(np.mean(pred_msi))

        data = {'Name': pids, 'Pred': preds,'GTs':gts}
        df = pd.DataFrame(data)
        df.to_excel(self.model_dir + "{}_{}_{}_{}_{}..xlsx".format(self.model_name, self.fp, self.op, self.lr, self.batch_size))

        fpr, tpr, thresholds = roc_curve(np.asarray(gts), np.asarray(preds), pos_label='MSI-H')
        auc = roc_auc_score(np.asarray(gts) == 'MSI-H', np.asarray(preds))
        print('auc=%f\n' % auc)

        print('automatic prediction done!!!!')

        return auc

    def test_model_external_temp(self):
        '''
        temp testing for efficiency testing

        the tiles belonging to the same patient are put under the same folder
        '''
        fkey = 'test2'
        data_transforms =transforms.Compose([
               transforms.Resize(224),
               transforms.ToTensor(),])

        image_datasets = ImageFolderWithPaths(os.path.join(self.test_dir, fkey), data_transforms)
        #image_datasets = ImageFolderWithPaths(os.path.join(self.test_dir, fkey), data_transforms[fkey])

        dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=self.batch_size,
                                                      shuffle=False, num_workers=self.num_workers)
        # dataset_sizes = {x: len(image_datasets[x]) for x in [fkey]}

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # build model
        model = self.build_model(2)
        model.to(device)
        model.load_state_dict(torch.load(self.model_dir +
                                             "{}_{}_{}_{}_{}.pt".format(self.model_name, self.fp, self.op, self.lr,
                                                                        self.batch_size)))
        model = nn.Sequential(model, nn.Softmax(dim=1))
        model.eval()

        imgs = []
        pred_tils = []
        for inputs, label, paths in dataloaders:
            inputs = inputs.to(device)

            with torch.no_grad():
                outputs = model(inputs)

                # imgs_n=[paths[i].split('\\')[-1] for i in range(len(paths))] #win10
            imgs_n = [paths[i].split('/')[-1] for i in range(len(paths))]  # linux
            preds_n = outputs.cpu().numpy()

            imgs.extend(imgs_n)
            pred_tils.extend(preds_n[:, 1])

        data = {'Name': imgs, 'Pred': pred_tils}
        df = pd.DataFrame(data)
        df.to_excel('debug1.xlsx')

        print('automatic prediction done!!!!')

    def test_end_to_end(self):
        '''
        end_to_end testing for easy usages
        input: path to wsi
        output: prediction masks saved in output path
        '''

        # s1: load model
        device = torch.device("cuda:" + str(self.cuda_id) if torch.cuda.is_available() else "cpu")
        model = self.load_model()
        model.eval()

        # s2:
        class_ind = self.class_interest
        wsis = sorted(os.listdir(self.wsi_path))
        thr=0.5 # 0.4->tils, 0.5->tumor
        wt=220  # blca: 210

        d_slides=[]          # save slide names
        d_grids=[]           # save tumor tile coordinates
        d_strides=[]          # save tile strides
        for img_name in wsis: # index12 for tils analysis paper example
            if self.wsi_ext in img_name:
                pid=img_name.split('.')[0]
                print("patient id is %s" % pid)
                if self.wsi_ext=='.czi':
                    wsi_tiling_pred_czi(self.wsi_path + img_name, self.output_path, img_name, self.tile_size, model,
                                    class_ind, device, thr)
                else:
                    grids,strides=wsi_tiling_pred(self.wsi_path+img_name,self.output_path,img_name,self.tile_size,model,class_ind,device,thr,wt)
                    d_slides.append(img_name)
                    d_grids.append(grids)
                    d_strides.append(strides)

        return d_slides,d_grids,d_strides

# ------------histroy--------------------------------------#
    # def test_end_to_end_ii(self):
    #     '''
    #     end_to_end testing for easy usages
    #     input: path to wsi
    #     output: prediction masks saved in output path
    #
    #     In this version, we assume that two models are sequentially applied on the wsi
    #     e.g., wsi -> tumor detector -> msi prediction
    #     '''
    #
    #     # s1: load model
    #     device = torch.device("cuda:" + str(self.cuda_id) if torch.cuda.is_available() else "cpu")
    #     model0 = self.load_model()
    #     model0.eval()
    #     class_ind0 = self.class_interest
    #
    #     # load the second model
    #     class_name2=['msi_h','mss']
    #     self.class_num=len(class_name2)
    #     self.model_dir='../../../data/kang_colon_data/msi_models/resnet18_torch/'
    #     self.model_name='resnet18'
    #     self.fp=0
    #     self.batch_size=4
    #     self.lr=0.0001
    #     self.op='adam'
    #     output_path1 = '../../../data/kang_colon_data/yonsei95_slide/msi_pred/'+self.output_path.split('/')[-2]+'/'
    #     model1=self.load_model()
    #     model1.eval()
    #     class_ind1=0
    #
    #     wsis = sorted(os.listdir(self.wsi_path))
    #     wsis_name=[]
    #     wsis_pred=[]
    #     wsis_gt=[]
    #     for img_name in wsis:
    #         if self.wsi_ext in img_name:
    #             pid=img_name.split('.')[0]
    #             print("patient id is %s" % pid)
    #             pred=wsi_tiling_pred_ii(self.wsi_path+img_name,self.output_path,img_name,self.tile_size,model0,class_ind0,device,
    #                                model1,output_path1,class_ind1)
    #
    #             wsis_name.append(pid)
    #             wsis_pred.append(pred)
    #             gt=self.output_path.split('/')[-2]
    #             wsis_gt.append(gt)
    #
    #     #data = {'Name': wsis_name, 'Pred': wsis_pred}
    #     #df = pd.DataFrame(data)
    #     #df.to_excel(output_path1[0]+'pred.xlsx')
    #     return wsis_name,wsis_pred,wsis_gt


    # def test_model_external_temp_tumor(self):
    #     '''
    #     can be used for external testing, all the images (unknow labels) are put under the same folder
    #
    #     each wsi has many tiles
    #
    #     output predictions and save results for each patient into a separate excel file
    #     '''
    #     data_transforms=transforms.Compose([
    #             transforms.Resize(224),
    #             transforms.ToTensor(),])
    #
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    #     # build model
    #     model = self.build_model(3)
    #     model.to(device)
    #     model.load_state_dict(torch.load(self.model_dir +
    #                                      "{}_{}_{}_{}_{}.pt".format(self.model_name, self.fp, self.op, self.lr,
    #                                                                 self.batch_size)))
    #     model = nn.Sequential(model, nn.Softmax(dim=1))
    #     model.eval()
    #
    #     wsis = sorted(os.listdir(self.wsi_path))
    #     for img_name in wsis[13:]:
    #         if self.wsi_ext in img_name:
    #             pid=img_name.split('.')[0]
    #             print("patient id is %s" % pid)
    #
    #             start_time = time.time()
    #
    #             image_datasets = TestDataset(self.test_dir, pid, data_transforms)
    #             # x,p=image_datasets.__getitem__(0)
    #
    #             dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=self.batch_size,
    #                                                   shuffle=False, num_workers=self.num_workers)
    #
    #             imgs = []
    #             pred_tumors = []
    #             for inputs, paths in dataloaders:
    #                 inputs = inputs.to(device)
    #
    #                 with torch.no_grad():
    #                     outputs = model(inputs)
    #
    #                     # imgs_n=[paths[i].split('\\')[-1] for i in range(len(paths))] #win10
    #                 imgs_n = [paths[i].split('/')[-1] for i in range(len(paths))]  # linux
    #                 preds_n = outputs.cpu().numpy()
    #
    #                 imgs.extend(imgs_n)
    #                 pred_tumors.extend(preds_n[:, 2])
    #
    #             data = {'Name': imgs, 'Pred': pred_tumors}
    #             df = pd.DataFrame(data)
    #             #df.to_excel(self.wsi_path+pid+'.xlsx')
    #             #df.to_excel(self.model_dir + pid + '.xlsx')
    #             df.to_excel(self.output_path + pid + '.xlsx')
    #
    #             print((time.time() - start_time) / 60)
    #
    #     print('automatic prediction done!!!!')