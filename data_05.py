'''
purpose:
generate a set of files with the same name as slides
so that GPU sever program knows paitent id for external testing

author: Hongming Xu, ccf, 2020
email: mxu@ualberta.ca
'''
import os

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


