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

rela_path='../../../'

data_path = rela_path+'data/pan_cancer_tils/models/resnet18/'
df=pd.read_excel(data_path+'logs.xlsx')
x=list(df['Training Time'])
y=list(df['Test Acc'])

data_path2 = rela_path+'data/pan_cancer_tils/models/shufflenet/'
df2=pd.read_excel(data_path2+'logs.xlsx')
x2=list(df2['Training Time'])
y2=list(df2['Test Acc'])

data_path3 = rela_path+'data/pan_cancer_tils/models/resnet34/'
df3=pd.read_excel(data_path3+'logs.xlsx')
x3=list(df3['Training Time'])
y3=list(df3['Test Acc'])

f,ax=plt.subplots(1)
plt.scatter(x[0:24],y[0:24],marker='o',alpha=0.7,cmap='tab20',label='Resnet18 (frozen 0% parameters)')
plt.scatter(x[24:48],y[24:48],marker='p',alpha=0.7,cmap='tab20',label='Resnet18 (frozen 80% parameters)')
plt.scatter(x2[0:24],y2[0:24],marker='s',alpha=0.7,cmap='tab20',label='Shufflenet (frozen 0% parameters)')
plt.scatter(x2[24:48],y2[24:48],marker='h',alpha=0.7,cmap='tab20',label='Shufflenet (frozen 80% parameters)')
plt.scatter(x3[0:24],y3[0:24],marker='^',alpha=0.7,cmap='tab20',label='Resnet34 (frozen 0% parameters)')
plt.scatter(x3[24:48],y3[24:48],marker='d',alpha=0.7,cmap='tab20',label='Resnet34 (frozen 80% parameters)')
#plt.scatter(x[18:24],y2[18:24],marker='p',alpha=0.7,cmap='tab20',label='Resnet18 (frozen 80% parameters, adam optimizer)')
plt.xlabel('Training time (minutes)')
plt.ylabel('Accuracy')
ax.legend()
#plt.ylim([0.650,0.83])
#plt.colorbar()  # show color scale
#plt.show()
plt.savefig('./at_tils.eps', format='eps')
plt.close()

# f=plt.figure()
# plt.plot(x,y,'bo',x2,y2,'rx',x3,y3,'g+')
# plt.xlabel('taining time (minutes)')
# plt.ylabel('testing accuracy')
# plt.legend(['resnet18','shufflenet','resnet34'])
# f.clear()
# plt.close()

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