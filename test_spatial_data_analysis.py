'''
purpose: learn spatial data analysis from website:https://pysal.org/esda/generated/esda.G_Local.html#esda.G_Local
'''
import pysal
import numpy as np
x,y=np.indices((5,5))
x.shape=(25,1)
y.shape=(25,1)
data=np.hstack([x,y])
wknn3=pysal.lib.weights.KNN(data,k=3)
print(wknn3.neighbors[0])


import pysal
import numpy as np
np.random.seed(10)
points=[(10,10),(20,10),(40,10),(15,20),(30,20),(30,30)]
w=pysal.lib.weights.DistanceBand(points,threshold=15)
print(w)

y=np.array([2,3,3.2,5,8,7])
lg=pysal.explore.esda.G_Local(y,w,transform='B')
print(lg.Zs)