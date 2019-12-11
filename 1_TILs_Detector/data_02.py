'''
code run for xx system on Hongming Xu's computer
purpose: divide into train, valid and test
'''
import os
import glob
import shutil
import concurrent.futures
from random import randrange
from itertools import repeat

ori_data=['../../../data/pan_cancer_tils/data_v01/tils/',
          '../../../data/pan_cancer_tils/data_v01/others/']

train_des=['../../../data/pan_cancer_tils/data_v02/train/tils/',
          '../../../data/pan_cancer_tils/data_v02/train/others/']

valid_des=['../../../data/pan_cancer_tils/data_v02/valid/tils/',
          '../../../data/pan_cancer_tils/data_v02/valid/others/']

test_des=['../../../data/pan_cancer_tils/data_v02/test/tils/',
          '../../../data/pan_cancer_tils/data_v02/test/others/']

for i in range(len(ori_data)):
    tp=ori_data[i]
    imgs = glob.glob(tp + '/*.png')
    trainn=round(len(imgs)*0.7)
    validn=round(len(imgs)*0.15)
    testn=len(imgs)-trainn-validn

    pids=set()
    for k in range(len(imgs)):
        #temp=imgs[k].split('\\') #win10
        temp = imgs[k].split('/')  # linux
        pid=temp[-1][0:12]
        pids.add(pid)      # add unique pid


    trn=0
    van=0
    ten=0
    for pid in pids:
        ind=randrange(3)

        imgs2 = glob.glob(tp + pid+'*.png')

        if ind==0 and trn<trainn:
            des_path=train_des[i]
            trn+=len(imgs2)
        elif ind==1 and van<validn:
            des_path=valid_des[i]
            van+=len(imgs2)
        elif ind==2 and ten<testn:
            des_path=test_des[i]
            ten+=len(imgs2)
        else:
            print('pay attentions....')
            des_path=train_des[i]
            trn+=len(imgs2)

        with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
            for _ in executor.map(shutil.copy,imgs2,repeat(des_path)):
                pass




