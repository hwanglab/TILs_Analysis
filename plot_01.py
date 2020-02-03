'''
ploting internal testing accuray over training times
for trained different models
purpose: select the best tils detector model
used only during developing study

author: Hongming Xu, CCF
email: mxu@ualberta.ca
'''

import matplotlib.pyplot as plt
import pandas as pd

data_path = '../../../data/pan_cancer_tils/models_v02/'
df=pd.read_excel(data_path+'logs.xlsx')
x=list(df['Training Time'])
y=list(df['Test Acc'])

data_path2 = '../../../data/pan_cancer_tils/models_v03/'
df2=pd.read_excel(data_path2+'logs.xlsx')
x2=list(df2['Training Time'])
y2=list(df2['Test Acc'])

data_path3 = '../../../data/pan_cancer_tils/models_v04/'
df3=pd.read_excel(data_path3+'logs.xlsx')
x3=list(df3['Training Time'])
y3=list(df3['Test Acc'])

f=plt.figure()
plt.plot(x,y,'bo',x2,y2,'rx',x3,y3,'g+')
plt.xlabel('taining time (minutes)')
plt.ylabel('testing accuracy')
plt.legend(['resnet18','shufflenet','resnet34'])
f.clear()
plt.close()

models=list(df['Models'])
ind=y.index(max(y))
print(f'best model is {models[ind]}\n')
print(f'best test acc is {y[ind]}\n')

models2=list(df2['Models'])
ind=y2.index(max(y2))
print(f'best shufflenet is {models2[ind]}\n ')
print(f'best test acc is {y2[ind]}\n')

models3=list(df3['Models'])
ind=y3.index(max(y3))
print(f'best resnet34 is {models3[ind]}\n ')
print(f'best test acc is {y3[ind]}\n')