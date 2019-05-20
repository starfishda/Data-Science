import numpy as np
from matplotlib import pyplot as plt

#수요일반
X = np.array([28,35,26,32,28,28,35,34,46,42,37])
Y = np.array([168, 170, 161, 167, 165, 183, 172, 175, 160, 189])
#월요일반 + 수요일반
#X = np.array([163, 177, 179, 168, 174, 176, 162, 172, 155, 157, 179, 155, 178, 165, 179, 163, 168, 170, 161, 167, 165, 183, 172, 175, 160, 189])
#Y = np.array([167,170,163,160,178,177,175,171,163,169,165,181,175,170,181,177,172,168,160,175,173,158,158,158,175,160])


#Mean
X_mean = np.mean(X)
Y_mean = np.mean(Y)
print('평균 : ',X_mean,Y_mean)

#Variance
X_var = np.var(X)
Y_var = np.var(Y)
print('분산 : ',X_var,Y_var)

#Standard Deviation
X_std = np.std(X)
Y_std = np.std(Y)
print('표준편차 : ',X_std,Y_std)

#Graph
plotData = [X,Y]
plt.boxplot(plotData)
plt.show();

