import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from IPython.display import Image
from sklearn.model_selection import cross_val_score



        # Reading the dataset:: the seperator of the datset is a semicolon ";"
dataset =pd.read_csv('dataset/student-por.csv', sep=';')
print(len(dataset))

# Dataset has 3 portuguese language grades: Grade1, Grade2, and Grade3
# We will simplfy the problem by generating a pass/fail binary columns (axis=1) based on the sum of the 3 grades
# If the sum >= 35, pass (=1) will be granded, else fail (=0) will be granded

dataset['pass']= dataset.apply(lambda row:1 if (row['G1'] + row['G2']+ row['G3']) >= 35 else 0, axis=1)

#Viewing the head(first 5 rows) of the dataset to check the new added column
print(dataset.head())

#Chekcing the number of columns in the dataset.
#We origanlly had 33 columns, now we have 34.
print(dataset.shape[1])

#Dropping the grades columns
dataset = dataset.drop(['G1','G2','G3'], axis=1)
print(dataset.head())
print(dataset.shape[1])

#Data statistics: Counting the total number of pass/fail in the dataset
nbr_pass=np.sum(dataset['pass'])
nbr_fail=len(dataset['pass']) - nbr_pass
per_pass=100*float(nbr_pass/len(dataset['pass']))

print('Data statistics')
print("%d students have passed out of %d students." % ( nbr_pass, len(dataset['pass'])))
print("Percentage of success is %.3f%%" % (per_pass))

#Generating Pie chart
data=[nbr_pass,nbr_fail]
labels=["Pass students", "Fail students"]
plt.pie(data, labels=labels, autopct="%1.1f%%")  # Add percents with one decimal
plt.title("Distribution of pass/fail in the dataset")
# plt.show()

#Data pre-processing : Transforming non-numerical values into numerical ones using the get_dummies function from pandas: hot-encoding in ML
#This process wil generate more coluumns in the dataset

dataset= pd.get_dummies(dataset, columns=['sex', 'school', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob','reason', 'guardian',
                                          'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
                                          'higher', 'internet', 'romantic'], 
                                          dtype=int)

print(dataset.head())
print(dataset.shape)



    #Split the dataset into a train set and a test set. 80/20 split ratio

#We will suffle the rows  
#sample is a function from the Pandas library commonly used to suffle the rows of a datatset
#By setting frac=1, you're instructing the function to return the entire datset
dataset = dataset.sample(frac=1)

#First 500 rows will be the train set. The rest (=149 rows) will be the test datatset
dataset_train = dataset[:500] 
print(dataset_train.shape)
dataset_test = dataset[500:]
print(dataset_test.shape)

# Dropping the pass column for both sets. We will save the dropped pass columns in a var
dataset_train_d = dataset_train.drop(['pass'],axis=1)
dataset_train_pass = dataset_train['pass']
dataset_test_d = dataset_test.drop(['pass'],axis=1)
dataset_test_pass = dataset_test['pass']
dataset_d=dataset.drop(['pass'], axis=1)
dataset_pass =dataset['pass']


#Training the decison tree ML model


# In the context of decision trees, max_depth refers to the maximum number of splits (or levels) allowed in a tree. It essentially controls the complexity of the tree:
# Lower max_depth: Results in a shallower tree with fewer splits. Shallower trees are generally less prone to overfitting the training data but might not capture intricate patterns in the data.

train_tree= tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
train_tree= train_tree.fit(dataset_train_d, dataset_train_pass)
dot_data = tree.export_graphviz(train_tree, out_file="performance.dot", label="all", impurity=False, proportion=True,
feature_names=list(dataset_train_d), class_names=["fail","pass"], filled=True, rounded=True)

#Checking the score of the tree against the testing set
score= train_tree.score(dataset_test_d,dataset_test_pass)
print(score)

#Cross validation 
scores= cross_val_score(train_tree,dataset_d,dataset_pass,cv=5)
print("Accuracy: %0.2f (+/-%0.2f)" % (scores.mean(), scores.std()*2))

#Trying different max_depth
depth_acc = np.empty((19,3), float)
i = 0
for max_depth in range(1, 20):
    t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
    scores_tree = cross_val_score(t, dataset_d, dataset_pass, cv=5)
    depth_acc[i,0] = max_depth
    depth_acc[i,1] = scores_tree.mean()
    depth_acc[i,2] = scores_tree.std() * 2
    i +=1
    print("Max depth: %d, Accuracy: %0.2f (+/- %0.2f)" % (max_depth, scores_tree.mean(), scores_tree.std() * 2))
print(depth_acc)


fig, ax = plt.subplots()
ax.errorbar(depth_acc[:,0], depth_acc[:,1], yerr=depth_acc[:,2]) 
plt.show()
#Saing a copy of the dataset 
# df = pd.DataFrame(dataset)
# df.to_csv("preprocessing.csv", index=False)


