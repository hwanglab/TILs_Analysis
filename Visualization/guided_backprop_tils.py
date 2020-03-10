"""
purpose: vanilla backprop visulazation for tils
reference: https://github.com/xhm1014/pytorch-cnn-visualizations

author: Hongming Xu, CCF
email: mxu@ualberta.ca
"""

import torch
from torch.nn import ReLU

import matplotlib.pyplot as plt

from utils_funcs import *


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        #first_layer = list(self.model.features._modules.items())[0][1]
        first_layer = self.model.conv1
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        #for pos, module in self.model.features._modules.items():
        layers=recursively_enumerate_model(self.model)
        for module in layers:
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward pass
        model_output = self.model(input_image)
        print(model_output)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.cuda.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        #one_hot_output[0][1]=1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.cpu().numpy()[0]
        return gradients_as_arr


if __name__ == '__main__':
    # Get params
    rela_path = '../../../'
    data_path = rela_path + 'data/pan_cancer_tils/data_v02/valid_clean_unbiased/others/'
    out_path = rela_path + 'data/pan_cancer_tils/visualization/others/'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_names = ['others', 'tils']
    model = build_model('resnet18', True, len(class_names), 0)
    model = model.to(device)
    model.load_state_dict(
        torch.load(rela_path + 'data/pan_cancer_tils/models/resnet18/' + 'resnet18_0_adam_0.0001_4.pt'))

    #model = nn.Sequential(model, nn.Softmax(dim=1))

    imgs = sorted(os.listdir(data_path))
    for t_img in imgs:
        original_image, prep_img, target_class = preprocess_image(data_path + t_img, device)

        # Guided backprop
        GBP = GuidedBackprop(model)

        model_output = GBP.model(prep_img)

        #model_output=model(prep_img)
        #print(model_output)

        # Get gradients
        guided_grads = GBP.generate_gradients(prep_img, target_class)


        # Save colored gradients
        save_gradient_images(guided_grads, out_path + t_img + '_Guided_BP_color')
        # Convert to grayscale
        grayscale_vanilla_grads = convert_to_grayscale(guided_grads)
        # Save grayscale gradients
        save_gradient_images(grayscale_vanilla_grads, out_path + t_img + '_Guided_BP_gray')


        # Positive and negative saliency maps
        #pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
        #save_gradient_images(pos_sal, file_name_to_export + '_pos_sal')
        #save_gradient_images(neg_sal, file_name_to_export + '_neg_sal')
        print('Guided backprop completed')
