import os
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import models
import torch.nn as nn
from torch.autograd import Variable
from torchsummary import summary

def img_to_var(pil_im,device,resize_im=True,crop_img=False):
    if resize_im:
        pil_im.thumbnail((224,224))
    if crop_img:
        width,height=pil_im.size
        new_width,new_height=224,224
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2

        # Crop the center of the image
        pil_im = pil_im.crop((left, top, right, bottom))

    im_as_arr=np.float32(pil_im)
    im_as_arr=im_as_arr.transpose(2,0,1)

    #for channel, _ in enumerate(im_as_arr):
    #    im_as_arr[channel] /= 255
    im_as_arr /= 255.0 # transforms.ToTensor() change the image pixels from [0,255]->[0,1] by dividing 255


    im_as_ten=torch.from_numpy(im_as_arr).float()
    im_as_ten.unsqueeze_(0)
    im_as_ten=im_as_ten.to(device)
    im_as_var=Variable(im_as_ten,requires_grad=True)

    return im_as_var

def preprocess_image(input_image,device):
    original_image=Image.open(input_image).convert('RGB')
    target_class=1 # tils
    prep_img=img_to_var(original_image,device,resize_im=False,crop_img=True)

    return original_image,prep_img,target_class

def recursively_enumerate_model(module):
    if list(module.children()) == []:
        return [module]
    else:
        enumerated_model = []
        for child in module.children():
            enumerated_model += recursively_enumerate_model(child)
        return enumerated_model

def freeze_weights(module):
    for param in module.parameters():
        param.requires_grad = False

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

def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr

def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)

def save_gradient_images(gradient, file_name):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    #if not os.path.exists('../results'):
    #    os.makedirs('../results')
    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    # Save image
    #path_to_file = os.path.join('../results', file_name + '.jpg')
    path_to_file = os.path.join(file_name + '.jpg')
    save_image(gradient, path_to_file)

def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale

    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im