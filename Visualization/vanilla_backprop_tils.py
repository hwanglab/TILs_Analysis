"""
purpose: vanilla backprop visulazation for tils
reference: https://github.com/xhm1014/pytorch-cnn-visualizations

author: Hongming Xu, CCF
email: mxu@ualberta.ca
"""

import matplotlib.pyplot as plt
from utils_funcs import *

class VanillaBackprop():
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        #first_layer = list(self.model.features._modules.items())[0][1]
        #first_layer = self.model.conv1
        first_layer = self.model[0].conv1
        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward
        model_output = self.model(input_image)
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.cuda.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        #one_hot_output=torch.FloatTensor(np.ones((1,model_output.size()[-1]),dtype=np.float32)) # added by Hongming
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.cpu().numpy()[0]
        return gradients_as_arr

if __name__ == '__main__':
    # Get params
    rela_path='../../../'
    data_path=rela_path+'data/pan_cancer_tils/data_v02/valid_clean_unbiased/tils/'
    out_path=rela_path+'data/pan_cancer_tils/visualization/tils/'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_names=['others','tils']
    model=build_model('resnet18',True, len(class_names),0)
    model=model.to(device)
    model.load_state_dict(torch.load(rela_path+'data/pan_cancer_tils/models/resnet18/'+'resnet18_0_adam_0.0001_4.pt'))
    model=nn.Sequential(model,nn.Softmax(dim=1))

    imgs=sorted(os.listdir(data_path))
    for t_img in imgs:
        original_image, prep_img, target_class=preprocess_image(data_path+t_img,device)


        #plt.imshow(np.transpose(prep_img.cpu().detach().numpy()[0, :, :, :], (1, 2, 0)).astype(np.uint8))
        #plt.show()

        # Vanilla backprop
        VBP = VanillaBackprop(model)
        # Generate gradients
        vanilla_grads = VBP.generate_gradients(prep_img, target_class)

        # Save colored gradients
        save_gradient_images(vanilla_grads, out_path+t_img + '_Vanilla_BP_color')
        # Convert to grayscale
        grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
        # Save grayscale gradients
        save_gradient_images(grayscale_vanilla_grads, out_path+t_img + '_Vanilla_BP_gray')
        print('Vanilla backprop completed')