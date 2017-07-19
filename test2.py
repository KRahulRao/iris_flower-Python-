import numpy as np

import pandas as pd

from sklearn.datasets import load_iris

from sklearn import tree

#test_idx = [0,50,100]



iris = load_iris()

#print iris.feature_names
#print iris.target_names




#print iris.data[0]

#print iris.target[0]


"""for i in range(len(iris.target)):

 print "Example %d: label %s, features %s" %(i,iris.target[i],iris.data[i]) """



data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

#data1.describe()


#Training Data

train_data =data1.drop(data1.index[0])
train_data =train_data.drop(data1.index[1])
train_data =train_data.drop(data1.index[2])
#train_target =data1.drop(data1.index[100])

#print train_data




#print type(test_data)


features=list(train_data.columns[:4])

Z=train_data["target"]
Y=train_data[features]

Y.values.reshape(1,-1)
Z.values.reshape(1,-1)


#Test Data
test_data = data1[features].iloc[0]
test_data=test_data.values
test_data.reshape(1,-1)

#print Z
#print Y

#Prediction Model
clf= tree.DecisionTreeRegressor()

clf=clf.fit(Y,Z)
print test_data
print "The Predicted Value is--------->   "  clf.predict(test_data)




"""
train_data = np.delete(iris.data,test_idx,axis=0);

train_target = train_target.reshape(1,-1)
train_data = train_data.reshape(1,-1)

#Testing Data

test_target =iris.target[test_idx]
test_data=iris.data[test_idx]


test_target = test_target.reshape(1,-1)
test_data = test_data.reshape(1,-1)

print train_target
print train_data


clf= tree.DecisionTreeRegressor()

clf=clf.fit(train_target,train_data)

print train_target.describe()

#print test_data

#print clf.predict(test_data)  """





