import numpy as np
from matplotlib import pyplot as plt

#수요일반
#height =np.array ([163, 177, 179, 168, 174, 176, 162, 172, 155, 157, 179, 155, 178, 165, 179, 163, 168, 170, 161, 167, 165, 183, 172, 175, 160, 189])
#weight = np.array([48, 66, 70, 63, 60, 70, 49, 70, 45, 66, 70, 56, 70, 44, 55, 49, 58, 72, 45, 57, 48, 90, 72, 73, 50, 75])
#월요일반 + 수요일반
height = np.array([163, 177, 179, 168, 174, 176, 162, 172, 155, 157, 179, 155, 178, 165,179, 163, 168, 170, 161, 167, 165, 183, 172, 175, 160, 189,167,170,163,160,178,177,175,171,163,169,165,181,175,170,181,177,172,168,160,175,173,158,158,158,175,160])
weight = np.array([48, 66, 70, 63, 60, 70, 49, 70, 45, 66, 70, 56, 70, 44, 55, 49, 58, 72, 45, 57, 48, 90, 72, 73, 50, 75, 55, 72, 54, 48, 64, 70, 79, 76, 60, 84, 57, 69, 66, 75, 72, 64, 71, 45, 55, 58, 73, 49, 47,50, 79, 56])

#Linear Regression
fit = np.polyfit(height,weight,1)
fit_fn = np.poly1d(fit)
print('Linear regression ', fit_fn)
print(fit)

#Graph
plt.plot(height,fit_fn(height),c='r')
plt.scatter(height, weight) 
plt.title("Ex1-a")
plt.xlabel("height")
plt.ylabel("weight")
plt.axis([145, 200, 40, 100])
plt.show() 
