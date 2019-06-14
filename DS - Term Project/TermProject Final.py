from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.preprocessing import minmax_scale
import numpy as np
import pandas as pd
import pydot
import graphviz
import random
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')

data = pd.read_csv('student-mat.csv', encoding = 'utf-8')

data['GAvg'] = (data['G1'] + data['G2'] + data['G3']) / 3

columns=list(data.columns.values)

# Create a list to store the data    
def define_grade(df):
    grades = []
    # For each row in the column,
    for row in df['GAvg']:
        # if more than a value,
        if row >= (0.9 * df['GAvg'].max()):
            # Append a letter grade
            grades.append('A')
        # else, if more than a value,
        elif row >= (0.7 * df['GAvg'].max()):
            # Append a letter grade
            grades.append('B')
        # else, if more than a value,
        elif row < (0.7 * df['GAvg'].max()):
            # Append a letter grade
            grades.append('C')   
    # Create a column from the list
    df['grades'] = grades

    return df

# Create tree using decision tree algorithm with DecisionTreeClassifier()
def learning(i, X_train, Y_train):
    global arr_predict
    learn = DecisionTreeClassifier(criterion = 'entropy')
    learn.fit(X_train, Y_train)
    #partitial decision tree    
    arr_predict[i] = learn.predict(X_test)
    
# Search max frequent value as index of each bagging data 
def modefinder(numbers):
    c = Counter(numbers) 
    maxcount = max(c.values())

    return [x_i for x_i, c in c.items() if c == maxcount]

#Convert Dirty Data into nan
for i in range(len(data)):
    if(data.iloc[i,0] != 'GP' and data.iloc[i,0] != 'MS'):
        data.iloc[i,0] = np.nan
    if(data.iloc[i,1] != 'F' and data.iloc[i,1] != 'M'):
        data.iloc[i,1] = np.nan
    if(data.iloc[i,2] < 15 or data.iloc[i,2] > 22):
        data.iloc[i,2] = np.nan
    if(data.iloc[i,3] != 'R' and data.iloc[i,3] != 'U'):
        data.iloc[i,3] = np.nan
    if(data.iloc[i,4] != 'GT3' and data.iloc[i,4] != 'LE3'):
        data.iloc[i,4] = np.nan
    if(data.iloc[i,5] != 'A' and data.iloc[i,5] != 'T'):
        data.iloc[i,5] = np.nan
    if(data.iloc[i,6] < 0 or data.iloc[i,6] > 4):
        data.iloc[i,6] = np.nan
    if(data.iloc[i,7] < 0 or data.iloc[i,7] > 4):
        data.iloc[i,7] = np.nan
    if(data.iloc[i,8] != 'at_home' and data.iloc[i,8] != 'health' and data.iloc[i,8] != 'other' and data.iloc[i,8] != 'services' and data.iloc[i,8] != 'teacher'):
        data.iloc[i,8] = np.nan
    if(data.iloc[i,9] != 'at_home' and data.iloc[i,9] != 'health' and data.iloc[i,9] != 'other' and data.iloc[i,9] != 'services' and data.iloc[i,9] != 'teacher'):
        data.iloc[i,9] = np.nan
    if(data.iloc[i,10] != 'course' and data.iloc[i,10] != 'other' and data.iloc[i,10] != 'home' and data.iloc[i,10] != 'reputation'):
        data.iloc[i,10] = np.nan
    if(data.iloc[i,11] != 'mother' and data.iloc[i,11] != 'father' and data.iloc[i,11] != 'other'):
        data.iloc[i,11] = np.nan
    if(data.iloc[i,12] < 1 or data.iloc[i,12] > 4):
        data.iloc[i,12] = np.nan
    if(data.iloc[i,13] < 1 or data.iloc[i,13] > 4):
        data.iloc[i,13] = np.nan
    if(data.iloc[i,14] < 0 or data.iloc[i,14] > 3):
        data.iloc[i,14] = np.nan
    if(data.iloc[i,15] != 'no' and data.iloc[i,15] != 'yes'):
        data.iloc[i,15] = np.nan
    if(data.iloc[i,16] != 'no' and data.iloc[i,16] != 'yes'):
        data.iloc[i,16] = np.nan
    if(data.iloc[i,17] != 'no' and data.iloc[i,17] != 'yes'):
        data.iloc[i,17] = np.nan
    if(data.iloc[i,18] != 'no' and data.iloc[i,18] != 'yes'):
        data.iloc[i,18] = np.nan
    if(data.iloc[i,19] != 'no' and data.iloc[i,19] != 'yes'):
        data.iloc[i,19] = np.nan
    if(data.iloc[i,20] != 'no' and data.iloc[i,20] != 'yes'):
        data.iloc[i,20] = np.nan
    if(data.iloc[i,21] != 'no' and data.iloc[i,21] != 'yes'):
        data.iloc[i,21] = np.nan
    if(data.iloc[i,22] != 'no' and data.iloc[i,22] != 'yes'):
        data.iloc[i,22] = np.nan
    if(data.iloc[i,23] < 1 or data.iloc[i,23] > 5):
        data.iloc[i,23] = np.nan
    if(data.iloc[i,24] < 1 or data.iloc[i,24] > 5):
        data.iloc[i,24] = np.nan
    if(data.iloc[i,25] < 1 or data.iloc[i,25] > 5):
        data.iloc[i,25] = np.nan
    if(data.iloc[i,26] < 1 or data.iloc[i,26] > 5):
        data.iloc[i,26] = np.nan
    if(data.iloc[i,27] < 1 or data.iloc[i,27] > 5):
        data.iloc[i,27] = np.nan
    if(data.iloc[i,28] < 1 or data.iloc[i,28] > 5):
        data.iloc[i,28] = np.nan
    if(data.iloc[i,29] < 0 or data.iloc[i,29] > 75):
        data.iloc[i,29] = np.nan
    if(data.iloc[i,30] < 0 or data.iloc[i,30] > 20):
        data.iloc[i,30] = np.nan
    if(data.iloc[i,31] < 0 or data.iloc[i,31] > 20):
        data.iloc[i,31] = np.nan
    if(data.iloc[i,32] < 0 or data.iloc[i,32] > 20):
        data.iloc[i,32] = np.nan

#Convert converted nan value into min value each column
for i in columns:
    data[i].fillna(data[i].value_counts().idxmin(),inplace=True)

#Normalization
data['minmax_absence'] = minmax_scale(data['absences'], axis = 0, copy = True)
data['mMax_G1'] = minmax_scale(data['G1'], axis = 0, copy = True)
data['mMax_G2'] = minmax_scale(data['G2'], axis = 0, copy = True)
data['mMax_G3'] = minmax_scale(data['G3'], axis = 0, copy = True)
data['mMax_GAvg'] = minmax_scale(data['GAvg'], axis = 0, copy = True)

#Show Outlier by histogram based on z-score of absence
plt.hist(data['minmax_absence'], bins=10, facecolor = 'blue', alpha = 0.5)
plt.title('Histogram of absence normalization')
plt.xlabel('Number of absence : Min Max Scale')
plt.ylabel('Number of student')
plt.show()

plt.hist(data['mMax_GAvg'], bins=10, facecolor = 'orange', alpha = 0.5)
plt.title('Histogram of GAvg normalization')
plt.xlabel('Number of GAvg : Min Max Scale')
plt.ylabel('Number of student')
plt.show()
        
#Make new column GAvg that is average of G1, G2, G3
data = define_grade(data)

# Map the data that has 'yes' or 'no' unique key
d = {'yes': 1, 'no': 0}
data['schoolsup'] = data['schoolsup'].map(d)
data['famsup'] = data['famsup'].map(d)
data['paid'] = data['paid'].map(d)
data['activities'] = data['activities'].map(d)
data['nursery'] = data['nursery'].map(d)
data['higher'] = data['higher'].map(d)
data['internet'] = data['internet'].map(d)
data['romantic'] = data['romantic'].map(d)

# Map the sex data
d = {'F': 1, 'M': 0}
data['sex'] = data['sex'].map(d)

# Map the school data
d = {'GP': 1, 'MS': 0}
data['school'] = data['school'].map(d)

# Map the address data
d = {'U': 1, 'R': 0}
data['address'] = data['address'].map(d)

# Map the famili size data
d = {'LE3': 1, 'GT3': 0}
data['famsize'] = data['famsize'].map(d)

# Map the parent's status
d = {'T': 1, 'A': 0}
data['Pstatus'] = data['Pstatus'].map(d)

# Map the parent's job
d = {'teacher': 0, 'health': 1, 'services': 2,'at_home': 3,'other': 4}
data['Mjob'] = data['Mjob'].map(d)
data['Fjob'] = data['Fjob'].map(d)

# Map the reason data
d = {'home': 0, 'reputation': 1, 'course': 2,'other': 3}
data['reason'] = data['reason'].map(d)

# Map the guardian data
d = {'mother': 0, 'father': 1, 'other': 2}
data['guardian'] = data['guardian'].map(d)

# Map the grades data
d = {'C': 0, 'B': 1, 'A': 2}
data['grades'] = data['grades'].map(d)

data.to_csv('data_categorical.csv')

# Show correlation school between grade average
model = linear_model.LinearRegression()
school_data = data['school'].to_numpy()
model.fit(school_data[:,np.newaxis], data['GAvg'])

plt.plot(school_data, model.predict(school_data[:, np.newaxis]), color = 'r')
plt.scatter(data['school'], data['GAvg'], color = 'b', alpha = 0.5)
plt.xlabel('School')
plt.ylabel('Grade Average')
plt.show()

# Show correlation age between grade average
age_data = data['age'].to_numpy()
model.fit(age_data[:,np.newaxis], data['GAvg'])

plt.plot(age_data, model.predict(age_data[:, np.newaxis]), color = 'r')
plt.scatter(data['age'], data['GAvg'], color = 'g', alpha = 0.5)
plt.xlabel('Age')
plt.ylabel('Grade Average')
plt.show()

#Remove Unnecassary Data from training feature
data.drop(["school","age"], axis=1, inplace=True)
student_features = data.columns.tolist()
student_features.remove('grades') 
student_features.remove('GAvg') 
student_features.remove('G1') 
student_features.remove('G2') 
student_features.remove('G3')

#Copy necassary data to x, y variable
x = data[student_features].copy()
y = data['grades'].copy()

#Split train data set and test data set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

#Define Decision tree classifier and train by using train set
grade_classifier = tree.DecisionTreeClassifier(criterion = 'entropy', max_leaf_nodes=len(x.columns), random_state=0)
grade_classifier.fit(x_train, y_train)

#Print accuracy of decision tree
predictions = grade_classifier.predict(x_test)
print('[Data Analysis Result]\n Decisioin Tree Accuracy : ', accuracy_score(y_test, predictions), '\n')

#Visualize Decision tree
dot_data = StringIO()  
tree.export_graphviz(grade_classifier, out_file='decisionTree.dot',  
                         feature_names=student_features, rounded = True, filled = True)  

with open('decisionTree.dot') as f:
    dot_graph = f.read()

dot = graphviz.Source(dot_graph)
dot.format = 'png'

#Data Evaluation
# Define original data to test data set 
X = data.drop('grades',axis = 1)
Y = data['grades']
X_test = X
Y_test = Y
arr_predict = pd.DataFrame()

# Shuffle Data
shuffle_data= data.sample(frac = 1).reset_index(drop = True)

#Read bagging dataset and define that to train dataset
bagging = int(len(data)/10)

for i in range(10):
    sample = shuffle_data.sample(frac = 0.1)
    X_train = sample.drop('grades',axis=1)
    Y_train = sample['grades']
    learning(i, X_train, Y_train)

pred_v=pd.Series(np.zeros(len(data)))

#Ensemble decision trees as majority(voting)
for i in range(len(data)):
    max_value = modefinder(arr_predict.iloc[i, :])
    pred_v.iloc[i] = max_value[0] 

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print('-------------------------------------------------\n')
print('[Data Evaluation Result]')
print('<Confusion Matrix>\n', confusion_matrix(y_test, y_pred))

print("Training Accuracy : ", model.score(x_train, y_train))
print("Testing Accuracy : ", model.score(x_test, y_test))

