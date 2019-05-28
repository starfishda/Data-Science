import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model

#Read file
data = pd.read_excel('DS-lab-3-dataset.xlsx', 'Dataset', index_col=None)
print(data)

#make array to numpy array & make height, weight array
dataset= np.array(data)
m_height = []
m_weight = []
f_height = []
f_weight = []
mw_ = []
fw_ = []
m_e = []
m_z = []
f_e = []
f_z = []

for i in dataset:
    if i[0] == 'Male':
        m_height.append(i[1])
        m_weight.append(i[2])
    else:
        f_height.append(i[1])
        f_weight.append(i[2])

m_height = np.array(m_height)
m_weight = np.array(m_weight)
f_height = np.array(f_height)
f_weight = np.array(f_weight)

#LinearRegression-Male
fit = np.polyfit(m_height,m_weight,1)
fit_fn = np.poly1d(fit)
plt.scatter(m_height,m_weight)
plt.plot(m_height,fit_fn(m_height),c='r')
plt.show()

#Calculate w' and input in weight_ array - Male
for i in dataset:
    mw_.append(fit_fn(i[1]))

mw_ = np.array(mw_)

#LinearRegression-Female
fit = np.polyfit(f_height,f_weight,1)
fit_fn = np.poly1d(fit)
plt.scatter(f_height,f_weight)
plt.plot(f_height,fit_fn(f_height),c='r')
plt.show()

#Calculate w' and input in m_weight_ array - Female
for i in dataset:
    fw_.append(fit_fn(i[1]))

fw_ = np.array(fw_)

#Calculate e array
count = 0
for i in  m_weight:
    m_e.append(i - mw_[count])
    count += 1

count = 0
for i in  f_weight:
    f_e.append(i - fw_[count])
    count += 1

m_e = np.array(m_e)
f_e = np.array(f_e)

me_mean = np.mean(m_e)
me_std = np.std(m_e)
fe_mean = np.mean(f_e)
fe_std = np.std(f_e)

#Calculate z array
for i in m_e:
    m_z.append((i-me_mean)/me_std)

for i in f_e:
    f_z.append((i-fe_mean)/fe_std)

m_z = np.array(m_z)
f_z = np.array(f_z)

#histogram about z
plt.hist(m_z)
plt.xticks([-2.0,-1.6,-1.2,-0.8,-0.4, 0, 0.4, 0.8, 1.2, 1.6, 2.0])
plt.show()

plt.hist(f_z)
plt.xticks([-2.0,-1.6,-1.2,-0.8,-0.4, 0, 0.4, 0.8, 1.2, 1.6, 2.0])
plt.show()

a = 0
count = 0
count1 = 0
total = 0

for i in dataset:
    if i[0] == 'Male':
        if m_z[count] < a:
            dataset[total,3] = 0
        elif m_z[count] >= a:
            dataset[total,3] = 5
        count+=1
    else:
        if f_z[count1] < a:
            dataset[total,3] = 0
        elif f_z[count1] >= a:
            dataset[total,3] = 5
        count1 +=1
    total += 1


    
dataset = pd.DataFrame(dataset,columns=['Gender','Height','Weight','BMI'])
dataset.to_excel('DS-lab-3-dataset-change-BMI-Gender.xlsx', sheet_name = 'Dataset')
