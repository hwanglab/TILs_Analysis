'''
purpose: plot figures for the tils paper

y:acc, x: training time
author: Hongming Xu
email: mxu@ualerta.ca
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_df=pd.read_excel('./logs.xlsx')
x=np.asarray(data_df['Training Time'].tolist())
y1=np.asarray(data_df['Valid Acc'].tolist())
y2=np.asarray(data_df['Test Acc'].tolist())


f,ax=plt.subplots(1)
plt.scatter(x[0:6],y2[0:6],marker='o',alpha=0.7,cmap='tab20',label='Resnet18 (frozen 0% parameters, sgd optimizer)')
plt.scatter(x[6:12],y2[6:12],marker='s',alpha=0.7,cmap='tab20',label='Resnet18 (frozen 0% parameters, adam optimizer)')
plt.scatter(x[12:18],y2[12:18],marker='^',alpha=0.7,cmap='tab20',label='Resnet18 (frozen 80% parameters, sgd optimizer)')
plt.scatter(x[18:24],y2[18:24],marker='p',alpha=0.7,cmap='tab20',label='Resnet18 (frozen 80% parameters, adam optimizer)')
plt.xlabel('Training time (minutes)')
plt.ylabel('Accuracy')
ax.legend(loc='lower right')
plt.ylim([0.980,1.002])
#plt.colorbar()  # show color scale
#plt.show()
plt.savefig('./at.eps', format='eps')
plt.close()

# sizes = 10*x
#
# f, ax = plt.subplots(1)
# #for i in ['Validation','Testing']:
# #    if i=='Validation':
# #        plt.scatter(x, y1, c=sizes, s=sizes, marker='o',alpha=0.5,
# #            cmap='tab20',label=i)
# #    else:
# plt.scatter(x, y2, c=sizes, marker='^',alpha=0.9,
#                     cmap='tab20',label='resent18')
# plt.xlabel('Training time (minutes)')
# plt.ylabel('Accuracy')
# ax.legend(loc='lower right')
# plt.ylim([0.971,1.002])
# #plt.colorbar()  # show color scale
# plt.show()
# plt.savefig('./at.eps', format='eps')
# plt.close()

# plt.figure(1)
# plt.plot(x,y1,'ro',x,y2,'bo')
# plt.xlabel('Training time (minutes)')
# plt.ylabel('Accuracy')
# plt.legend(['Validation','Testing'])
# plt.show()
# plt.clear()
# plt.close()