'''
code run for win10 system on Hongming Xu's computer
purpose: count how many tils patch from public tcga
'''

import os
import glob
import shutil

ori_path='Y:/projects/tcga_til_analysis/TIL_patches/training_data_multi_zip/lym_cnn_training_data/'

des_path=['Y:/projects/data/pan_cancer_tils/data_v01/tils/','Y:/projects/data/pan_cancer_tils/data_v01/others/']

subfolders=[f.path for f in os.scandir(ori_path) if f.is_dir()]
print('the number of folders is=%d\n' % len(subfolders))

total_num=0 # total number of image patches
total_til=0
total_others=0
k=0
for t in range(len(subfolders)):
    temp_path=subfolders[t]
    imgs=glob.glob(temp_path+'/*.png')
    total_num+=len(imgs)
    print(temp_path)
    try:
        file=open(temp_path+"/label.txt","r")
        labels=file.readlines()
        file.close()
    except:
        print('no label file~~~~~')
        break

    for i in range(len(labels)):
        plabel=labels[i].split(' ')[1]
        img_label=labels[i].split(' ')[2]
        pid=[item for item in img_label.split('.') if len(item)==23 or len(item)==24]
        if plabel=='1':
            total_til+=1
            shutil.copy(temp_path+'/'+labels[i].split(' ')[0],des_path[0]+pid[0]+'_'+labels[i].split(' ')[0])
            k+=1 # control copy the same number of non-tils patches
            print(pid[0]+'_'+labels[i].split(' ')[0])


        elif plabel=='0' or plabel=='-1':
            total_others+=1
            if k>0:
                shutil.copy(temp_path + '/' + labels[i].split(' ')[0],
                            des_path[1] + pid[0] + '_' + labels[i].split(' ')[0])
                k-=1
                print(pid[0] + '_' + labels[i].split(' ')[0])

        else:
            print('unexpected label!!!!!!!!!!!!')
            break


print('total number of image patches is %d\n' % total_num)
print('the number of tils is %d\n' % total_til)
print('the number of non-tils is %d\n' % total_others)