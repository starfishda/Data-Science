import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model

#Wendsday
#height =np.array ([163, 177, 179, 168, 174, 176, 162, 172, 155, 157, 179, 155, 178, 165, 179, 163, 168, 170, 161, 167, 165, 183, 172, 175, 160, 189])
#weight = np.array([48, 66, 70, 63, 60, 70, 49, 70, 45, 66, 70, 56, 70, 44, 55, 49, 58, 72, 45, 57, 48, 90, 72, 73, 50, 75])
#Monday + Wendsday
height = np.array([163, 177, 179, 168, 174, 176, 162, 172, 155, 157, 179, 155, 178, 165,179, 163, 168, 170, 161, 167, 165, 183, 172, 175, 160, 189,167,170,163,160,178,177,175,171,163,169,165,181,175,170,181,177,172,168,160,175,173,158,158,158,175,160])
weight = np.array([48, 66, 70, 63, 60, 70, 49, 70, 45, 66, 70, 56, 70, 44, 55, 49, 58, 72, 45, 57, 48, 90, 72, 73, 50, 75, 55, 72, 54, 48, 64, 70, 79, 76, 60, 84, 57, 69, 66, 75, 72, 64, 71, 45, 55, 58, 73, 49, 47,50, 79, 56])

reg = linear_model.LinearRegression()
reg.fit(height[:, np.newaxis], weight)
px = np.array([height.min()-1, height.max() + 1])
py = reg.predict(px[:, np.newaxis])
plt.scatter(height, weight)
plt.plot(px,py,color='r')
plt.show()
