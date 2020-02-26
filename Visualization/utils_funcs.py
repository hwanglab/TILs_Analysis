import os
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import models
import torch.nn as nn
from torch.autograd import Variable

def img_to_var(pil_im,resize_im=True):
    if resize_im:
        pil_im.thumbnail((224,224))
    im_as_arr=np.float32(pil_im)
    #im_as_arr=im_as_arr.transpose(2,0,1)
    im_as_ten=torch.from_numpy(im_as_arr).float()
    im_as_ten.unsqueeze_(0)
    im_as_var=Variable(im_as_ten,requires_grad=True)

    return im_as_var

def preprocess_image(input_image):
    original_image=Image.open(input_image).convert('RGB')
    target_class=1 # tils
    prep_img=img_to_var(original_image)

    return original_image,prep_img,target_class

def recursively_enumerate_model(module):
    if list(module.children()) == []:
        return [module]
    else:
        enumerated_model = []
        for child in module.children():
            enumerated_model += recursively_enumerate_model(child)
        return enumerated_model

def build_model(model_name,img_init,n_classes,fp):
    if model_name == 'resnet18':
        model_ft = models.resnet18(pretrained=img_init)
    elif model_name =='shufflenet':
        model_ft = models.shufflenet_v2_x1_0(pretrained=img_init)
    elif model_name == 'resnet34':
        model_ft = models.resnet34(pretrained=img_init)
    else:
        raise ValueError('unrecognized model version: {}'.format(model_name))

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, n_classes)

    layers = recursively_enumerate_model(model_ft)
    layers = [layer for layer in layers if (type(layer) != nn.BatchNorm2d and len(list(layer.parameters())) > 0)]
    for layer in layers[:round(fp * len(layers))]:
        freeze_weights(layer)

    #    raise ValueError('incorrect mode setting: {}'.format(mode))

    return model_ft
