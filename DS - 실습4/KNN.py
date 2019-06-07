import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize

#Read file
data = pd.read_csv('knn_data.csv')

#Type K num
K = int(input('Type K num : '))

#Normalization dataset
def normalization(data):
    data.iloc[:,0] = (data.iloc[:,0]- data.iloc[:,0].mean()) / (data.iloc[:,0].max() - data.iloc[:,0].min())
    return data

#Caculate distance of each node
def distance(data_learn): 
    count = 0
    for i in range(len(data_learn) - 1):
        data.iloc[count,-1] =((data_learn.iloc[count, 0] - point_x)**2 + (data_learn.iloc[count, 1] - point_y)**2)**0.5
        count+=1

#Predict output of testing data
def check_predict(train_data, output):
    train_data = train_data.sort_values(by = "distance" , ascending = True)
    count = 0;
    
    for i in range(K):
        if(train_data.iloc[i,2] == output):
            count += 1

    if count > (K / 3):
        return 1
    else:
        return 0

#Main    
def learning(train_data, testing_data):
    train_data["distance"] = 100
    accuracy = 0
    for j in range(len(testing_data)):
        count = 0
        for i in range(len(train_data)):
            train_data.iloc[count,-1] =((train_data.iloc[count, 0] - testing_data.iloc[j,0])**2 + (train_data.iloc[count, 1] - testing_data.iloc[j,1])**2)**0.5
            count +=1
        accuracy += check_predict(train_data, testing_data.iloc[j,-1])

    accuracy = (accuracy/len(testing_data)) * 100
    return accuracy

#Split data to test and train
def data_split(data, test_radio):
    size = int(len(data) * test_radio)
    data = data.sample(frac=1).reset_index(drop=True)
    test1 = data.iloc[: size]
    test2 = data.iloc[size : size*2]
    test3 = data.iloc[size*2 : size*3]
    test4 = data.iloc[size*3 : size*4]
    test5 = data.iloc[size*4 : ]
    return test1, test2, test3, test4, test5


#Print
data_learn = normalization(data)
testdata1, testdata2, testdata3, testdata4, testdata5 = data_split(data_learn, 0.2)
print("1st Data Set: \n",testdata1,"\n")
print("2nd Data Set: \n",testdata2,"\n")
print("3rd Data Set: \n",testdata3,"\n")
print("4th Data Set: \n",testdata4,"\n")
print("5th Data Set: \n",testdata5,"\n")

a = learning(pd.concat([testdata2, testdata3, testdata4, testdata5]), testdata1)
print("\n1st Accuracy : ",a,"%\n")
b = learning(pd.concat([testdata1, testdata3, testdata4, testdata5]), testdata2)
print("2nd Accuracy : ",b,"%\n")
c = learning(pd.concat([testdata1, testdata2, testdata4, testdata5]), testdata3)
print("3rd Accuracy : ",c,"%\n")
d = learning(pd.concat([testdata1, testdata2, testdata3, testdata5]), testdata4)
print("4th Accuracy : ",d,"%\n")
e = learning(pd.concat([testdata1, testdata2, testdata3, testdata4]), testdata5)
print("5th Accuracy : ",e,"%\n")
total = (a + b + c + d + e) / 5
print("Total Accuracy(Average) : ",total,"%")
