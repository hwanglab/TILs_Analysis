'''
purpose:
generate a set of files with the same name as slides
so that GPU sever program knows paitent id for external testing

author: Hongming Xu, ccf, 2020
email: mxu@ualberta.ca
'''
import os
import pandas as pd

czi_stomach=False
lee_colon=True

if czi_stomach==True:
    ## 1) for the .czi images
    wsi_path=['Z:/Datasets/Stomach_Cancer_Stage_4_Immunotherapy_Slide_image_Yonsei_JaeHo_Cheong/Stomach_Cancer_Stage4_Immunotherapy/biopsy_45pts/',
              'Z:/Datasets/Stomach_Cancer_Stage_4_Immunotherapy_Slide_image_Yonsei_JaeHo_Cheong/Stomach_Cancer_Stage4_Immunotherapy/surgical_19pts/']

    dest_path=['../../data/Stomach_Cancer_Stage4_Immunotherapy/biopsy_45pts/',
               '../../data/Stomach_Cancer_Stage4_Immunotherapy/surgical_19pts/']

    wsi_ext='.czi'
    for tt in range(len(wsi_path)):
        temp_path=wsi_path[tt]
        d_path=dest_path[tt]
        slides=sorted(os.listdir(temp_path))
        for i in range(len(slides)):
            if '.czi' in slides[i]:
                with open(d_path+slides[i]+'.txt',"w") as file:
                    file.write('create a blank file')

if lee_colon==True:
    ## 2) for lee_colon images
    wsi_path='../../data/Colon_St_Mary_Hospital_SungHak_Lee_Whole_Slide_Image/CRC St. Mary hospital/'

    dest_path='../../data/lee_colon_data/wsi_tumor_files/'

    wsi_ext='.tiff'

    clinic_info = pd.read_excel('../../data/lee_colon_data/Colorectal cancer dataset.xlsx')
    pid = [i + j for i, j in zip(clinic_info['S no (primary)'].tolist(), clinic_info['Sub no (T)'].tolist())]
    pid2 = [sub.replace('#', '-') for sub in pid if isinstance(sub, str)]

    wsis = sorted(os.listdir(wsi_path))
    for img_name in wsis:
        if wsi_ext in img_name:
            temp_split = img_name.split('-')
            temp_split[1] = temp_split[1].zfill(6)
            pp = '-'.join(temp_split)
            if pp[:-5] in pid2:
                with open(dest_path + img_name + '.txt', "w") as file:
                    file.write('create a blank file')


