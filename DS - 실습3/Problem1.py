import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model

#Read file
data = pd.read_excel('DS-lab-3-dataset.xlsx', 'Dataset', index_col=None)
print(data)

#make array to numpy array & make height, weight array
dataset= np.array(data)
height = []
weight = []
weight_ = []
e = []
z = []

for i in dataset:
    height.append(i[1])
    weight.append(i[2])

height = np.array(height)
weight = np.array(weight)

#LinearRegression
fit = np.polyfit(height,weight,1)
fit_fn = np.poly1d(fit)
plt.scatter(height,weight)
plt.plot(height,fit_fn(height),c='r')
plt.show()

#Calculate w' and input in weight_ array
for i in dataset:
    weight_.append(fit_fn(i[1]))

weight_ = np.array(weight_)

fit = np.polyfit(height,weight_,1)
fit_fn = np.poly1d(fit)
plt.scatter(height,weight_)
plt.plot(height,fit_fn(height),c='r')
plt.show()

#Calculate e array
count = 0
for i in  weight:
    e.append(i - weight_[count])
    count += 1

e = np.array(e)

e_mean = np.mean(e)
e_std = np.std(e)

#Calculate z array
for i in e:
    z.append((i-e_mean)/e_std)

z = np.array(z)

#histogram about z
plt.hist(z)
plt.xticks([-2.0,-1.6,-1.2,-0.8,-0.4, 0, 0.4, 0.8, 1.2, 1.6, 2.0])
plt.show()

a = 0
count = 0

for i in z:
    if i < a:
        dataset[count,3] = 0
    else:
        dataset[count,3] = 5
    count+=1
    
dataset = pd.DataFrame(dataset,columns=['Gender','Height','Weight','BMI'])
dataset.to_excel('DS-lab-3-dataset-change-BMI.xlsx', sheet_name = 'Dataset')





    
