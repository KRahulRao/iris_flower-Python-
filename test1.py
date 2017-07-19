from sklearn  import tree

features =[[140,"1"],[150,"1"],[160,"1"],[170,"1"],[140,"0"],[130,"0"],
[120,"0"],[110,"0"]]

labels =["1","1","1","0","0","1","0","1"]

clf= tree.DecisionTreeClassifier()

clf= clf.fit(features,labels)

print clf.predict([[140,"0"],[150,"1"]])
