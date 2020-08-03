'''
purpose: try different thresholds on tils prediction maps
'''
import matplotlib.pyplot as plt
from PIL import Image

img_path='../../../data/kang_colon_slide/colon_IHC_JK/til_maps/'
til_pred=plt.imread(img_path+'HE_gray.png')
color_pred=plt.imread(img_path+'HE_color.png')

color_pred[:,:,2]=color_pred[:,:,2]*255.0

thr=[0.6,0.7,0.8,0.9]
for t in thr:
    temp=(til_pred[:,:,0]>t)
    color_pred[:,:,0]=temp*255.0

    img = Image.fromarray(color_pred.astype('uint8'))
    img.save(img_path + 'HE_color' + '_' + str(t)+'.png')
t=0