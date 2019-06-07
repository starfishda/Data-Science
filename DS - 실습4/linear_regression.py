import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random

#Read file
data = pd.read_csv('linear_regression_data.csv')

#Split data to training data & testing data by random
def train_test_split(dataset):

    list = []
    size = len(dataset)
    a = int(len(dataset) * 0.8)
    
    training_data = dataset.iloc[:a].reset_index(drop=True)
    
    for i  in range(a):
        num = random.randrange(0,size)
        while num in list:
            num = random.randrange(0,size)
        list.append(num)

    list.sort(reverse=True)

    for j in range(a):
        training_data.loc[j] = dataset.iloc[list[j]]
        dataset =  dataset.drop(list[j],0)

    testing_data = dataset.iloc[:].reset_index(drop=True)
    return training_data,testing_data

#print trainind data & testing data
training_data = train_test_split(data)[0]
print("Training Dataset : \n",training_data,"\n")
testing_data = train_test_split(data)[1]
print("Testing Dataset : \n",testing_data,"\n")

#Linear Regression
fit = np.polyfit(training_data.iloc[:,0],training_data.iloc[:,1],1)
fit_fn = np.poly1d(fit)
print('E = ',fit_fn,"\n")

#RSS
predict = fit_fn(testing_data.iloc[:,0]) #predict to use E
predict = np.array(predict)
output = np.array(testing_data.iloc[:,1]) #Real output of testing set
print('Predict of testing data = ' , predict,"\n")
print('Output of testing data = ', output,"\n")
residual = output - predict
print('Residual of testing data = ',residual,"\n")
RSS = sum(residual*residual)
print('RSS = ',RSS)

#Graph
plt.scatter(testing_data.iloc[:,0],testing_data.iloc[:,1])
plt.plot(training_data.iloc[:,0], fit_fn(training_data.iloc[:,0]),c='r')
plt.show()


