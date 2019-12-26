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

f=plt.figure()
plt.plot(x,y,'bo',x2,y2,'ro')
plt.xlabel('taining time (minutes)')
plt.ylabel('testing accuracy')
plt.legend(['resnet18','shufflenet'])
f.clear()
plt.close()

models=list(df['Models'])
ind=y.index(max(y))
print(f'best model is {models[ind]}\n')
print(f'best test acc is {y[ind]}\n')