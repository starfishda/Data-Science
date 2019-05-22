import numpy as np
from io import StringIO
from matplotlib import pyplot as plt

data = np.genfromtxt('DS-minilab-2-dataset.csv', delimiter=',', dtype="|U5")
print(data)
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

#Linear Regression
ofit = np.polyfit(ohlist,owlist,1)
ofit_fn = np.poly1d(ofit)
        
fit = np.polyfit(mhlist,mwlist,1)
fit_fn = np.poly1d(fit)

ffit = np.polyfit(fhlist,fwlist,1)
ffit_fn = np.poly1d(ffit)

#Graph
plt.figure(1)
plt.scatter(ohlist,owlist)
plt.plot(ohlist,ofit_fn(ohlist),c='r')
plt.show()

plt.figure(2)
plt.scatter(mhlist,mwlist)
plt.plot(mhlist, fit_fn(mhlist),c='r')
plt.show()

plt.figure(3)
plt.scatter(fhlist,fwlist)
plt.plot(fhlist, ffit_fn(fhlist),c='r')
plt.show()
