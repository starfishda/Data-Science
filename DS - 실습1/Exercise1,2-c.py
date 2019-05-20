import numpy as np
from matplotlib import pyplot as plt

#Make randon array
data = np.random.randint(10 , size=10000)
temp = np.array([0,0,0,0,0,0,0,0,0,0])

#Count
for i in data:
    temp[i] +=1

#Print Result
for i in range(0,10):
    print(i, '의 개수 : ',temp[i])
    

#Graph
langs = ['0','1','2','3','4','5','6','7','8','9']
num = [temp[0],temp[1],temp[2],temp[3],temp[4],temp[5],temp[6],temp[7],temp[8],temp[9]]
plt.pie(num, labels = langs, autopct='%1.2f%%')
plt.show()    

