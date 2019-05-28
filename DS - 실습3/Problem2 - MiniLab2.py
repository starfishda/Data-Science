import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model

#Read file
df = pd.read_csv('DS-minilab-2-dataset.csv')
print(df)

#Change to numpy array and fill missing data to 0
data = np.array(df.fillna(value = 0))
print(data)

#Set variance
maleHeight = 0
maleWeight = 0
malecount = 0

femaleHeight = 0
femaleWeight = 0
femalecount = 0

ohlist = []
owlist = []
mhlist = []
mwlist = []
fhlist = []
fwlist = []
        

# Check 'Male' and 'Female' value and calculate average
for i in data[1:,:]:
    if(i[0] == 'Male'):
        maleHeight += int(i[1])
        mhlist.append(int(i[1]))
        maleWeight += int(i[2])
        mwlist.append(int(i[2]))
        malecount += 1
    else:
        femaleHeight += int(i[1])
        fhlist.append(int(i[1]))
        femaleWeight += int(i[2])
        fwlist.append(int(i[2]))
        femalecount += 1
    ohlist.append(int(i[2]))
    owlist.append(int(i[1]))

maleheightmean = maleHeight / malecount
maleweightmean = maleWeight / malecount
femaleheightmean = femaleHeight / femalecount
femaleweightmean = femaleWeight / femalecount

#make list to put average value
count = 0;
for i in mhlist:
        if(i == 0):
                mhlist[count] = int(maleheightmean)
        count += 1

count = 0;
for i in mwlist:
        if(i == 0):
                mwlist[count] = int(maleweightmean)
        count += 1

count = 0;
for i in fhlist:
        if(i == 0):
                fhlist[count] = int(femaleheightmean)
        count += 1

count = 0;
for i in fwlist:
        if(i == 0):
                fhlist[count] = int(femaleheightmean)
        count += 1


# Make numpy array
ohlist = np.array(ohlist)
owlist = np.array(owlist)
mhlist = np.array(mhlist)
mwlist = np.array(mwlist)
fhlist = np.array(fhlist)
fwlist = np.array(fwlist)

reg = reg = linear_model.LinearRegression()

# Show original model
reg.fit(ohlist[:, np.newaxis], owlist)
px = np.array([ohlist.min()-1, ohlist.max() + 1])
py = reg.predict(px[:,np.newaxis])
plt.scatter(ohlist, owlist)
plt.plot(px,py,color='r')
plt.show()

# Show change Male model
reg.fit(mhlist[:, np.newaxis], mwlist)
px = np.array([mhlist.min()-1, mhlist.max() + 1])
py = reg.predict(px[:,np.newaxis])
plt.scatter(mhlist, mwlist)
plt.plot(px,py,color='r')
plt.show()

# Show change Female model
reg.fit(fhlist[:, np.newaxis], fwlist)
px = np.array([fhlist.min()-1, fhlist.max() + 1])
py = reg.predict(px[:,np.newaxis])
plt.scatter(fhlist, fwlist)
plt.plot(px,py,color='r')
plt.show()
