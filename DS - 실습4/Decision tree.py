import pandas as pd
import numpy as np
import random

#Read file
data = pd.read_csv('decision_tree_data.csv')

#Calculate entropy num
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts = True)
    entropy = -np.sum([(counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    return entropy

#Calculate each Information Gain
def InfoGain(data, split_attribute_name, target_name):

    #Calculate the entropy of the total dataset
    total_entropy = entropy(data[target_name])
    print('Entropy(D) = ', total_entropy)

    #Calculate the weighted entropy
    vals, counts = np.unique(data[split_attribute_name], return_counts = True)
    Weighted_Entropy = np.sum([(counts[i] / np.sum(counts)) * entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name])
                               for i in range(len(vals))])
    print('Child Entropy(', split_attribute_name, ') = ', Weighted_Entropy)

    #Calculate the information gain
    Information_Gain = total_entropy - Weighted_Entropy
    print('Information Gain(',split_attribute_name,') = ',Information_Gain,'\n')
    return Information_Gain


#Make tree
def ID3(data,originaldata,features,target_attribute_name,parent_node_class = None):

    #If all target_values have the same value, return this value
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
 
    #If the dataset is empty, return the mode target feature value in the original dataset
    elif len(data)==0:
        return np.unique(originaldata[target_attribute_name])\
               [np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]
 
    #Feature Empty -> return parent node class
    elif len(features) ==0:
        return parent_node_class
 
    #Grow the tree
    else:
        # The mode target feature value of the current node
        parent_node_class = np.unique(data[target_attribute_name])\
                            [np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        
        # Select the feature which best splits the dataset
        item_values = [InfoGain(data,feature,target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        
        # Create the tree structure. The root gets the name of the feature (best_feature) with the maximum information
        tree = {best_feature:{}}
        # Remove the feature with the best 
        features = [i for i in features if i != best_feature]
        
        # Grow a branch under the root node
        for value in np.unique(data[best_feature]):
            # Split the dataset
            sub_data = data.where(data[best_feature] == value).dropna()
            
            # ID3
            subtree = ID3(sub_data,data,features,target_attribute_name,parent_node_class)
            tree[best_feature][value] = subtree
            
        return tree
            
#Predict when input training data into tree
def predict(query,tree,default = 1):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]] 
            except:
                return default
  
            result = tree[key][query[key]]
            if isinstance(result,dict):
                return predict(query,result)
            else:
                return result

#Split data to test and training
def train_test_split(dataset):

    list = []
    size = len(dataset)
    a = int(len(dataset) * 0.9)
    
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

training_data = train_test_split(data)[0]
print("Training Dataset : \n",training_data,"\n")
testing_data = train_test_split(data)[1]
print("Testing Dataset : \n",testing_data,"\n")

def test(data,tree):
    #Create new query instances by simply removing the target feature column from the original dataset and 
    #convert it to a dictionary
    queries = data.iloc[:,:-1].to_dict(orient = "records")
    
    #Create a empty DataFrame in whose columns the prediction of the tree are stored
    predicted = pd.DataFrame(columns=["predicted"]) 
    
    #Calculate the prediction accuracy
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(queries[i],tree,1.0)
          
    if( (np.sum(predicted["predicted"] == data["interview"]) != 0)):
         print('Accuracy : ',(np.sum(predicted["predicted"] == data["interview"])/len(testing_data))*100,'%')
    else:
        print("Accuracy : 0%")
        
from pprint import pprint    
tree = ID3(training_data,training_data,training_data.columns[:-1],training_data.columns[-1])
pprint(tree)
test(testing_data,tree)
