from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

#reading traing data from wine train.csv
wine_training = pd.read_csv(r'C:\Users\Vinni\Desktop\Sem 3 Spring 2021\CS 529 ML\Prog1 Decision Trees\Wine\wine-train.csv',header=0)
#0th column has the index and first 11 columns of the csv have the data and the last column is the class or category hence, selecting the correct columns from the csv file
X_train = wine_training.iloc[:, 1:12]
Y_train = wine_training.iloc[:, -1:]

#readin testing data from wine-test.csv
wine_testing = pd.read_csv(r'C:\Users\Vinni\Desktop\Sem 3 Spring 2021\CS 529 ML\Prog1 Decision Trees\Wine\wine-test.csv',header=0)
#selecting the appropriate columns from wine test csv file
X_test = wine_testing.iloc[:, 1:12]
Y_test = wine_testing.iloc[:, -1:]

#Decision Tree
#Decision tree with default parameters
tree = DecisionTreeClassifier(random_state=1)
classifier = tree.fit(X_train, Y_train)
print("Decision Tree classifier with default parameters")
print('Accuracy for training data: %.3f'%tree.score(X_train, Y_train))
print('Accuracy for testing data: %.3f'%tree.score(X_test, Y_test))

#Decision tree with changed parameters
tree = DecisionTreeClassifier(random_state=1, ccp_alpha=0.0, min_impurity_decrease=0, min_samples_leaf=90,min_samples_split=200,max_leaf_nodes=50, max_depth=20)
classifier = tree.fit(X_train, Y_train)
print("Decision Tree classifier with changed parameters")
print('Accuracy for training data: %.3f'%tree.score(X_train, Y_train))
print('Accuracy for testing data: %.3f'%tree.score(X_test, Y_test))

#RANDOM FOREST
#Random Forest with Default parameters
y=np.ravel(Y_train)
clf = RandomForestClassifier(random_state=1)
clf.fit(X_train, y)
print("Accuracy using Random Forest Default Parameters")
print('Accuracy Training Data: %.3f'%clf.score(X_train, Y_train))
print('Accuracy Testing Data: %.3f'%clf.score(X_test, Y_test))

#Random Forest with changed parameters
y=np.ravel(Y_train)
clf = RandomForestClassifier(criterion='gini',random_state=10, max_depth=22, n_estimators=1000, ccp_alpha=0.0)
clf.fit(X_train, y)
print("Accuracy using Random Forest changed parameters")
print('Accuracy Training Data: %.3f'%clf.score(X_train, Y_train))
print('Accuracy Testing Data: %.3f'%clf.score(X_test, Y_test))

#KAGGLE CODE WINE
wine_kaggle = pd.read_csv(r'C:\Users\Vinni\Desktop\Sem 3 Spring 2021\CS 529 ML\Prog1 Decision Trees\Wine\wine-kaggle.csv', header=None)
ktree = RandomForestClassifier(criterion='gini',random_state=10, max_depth=22,n_estimators=1000, ccp_alpha=0.0)
#0.64800 WITH ktree = RandomForestClassifier(criterion='gini',random_state=10, max_depth=22,n_estimators=1000, ccp_alpha=0.0)
y=np.ravel(Y_train)
X_kaggle = wine_kaggle.iloc[:, 1:12]
ktree.fit(X_train, y)
kaggle_prediction=ktree.predict(X_kaggle)
#forming a datafrme to save the predictions into a csv file with id and class as headers
frame = pd.DataFrame(kaggle_prediction, columns=['class'])
frame.index=frame.index+1
frame.to_csv(r'C:\Users\Vinni\Desktop\Sem 3 Spring 2021\CS 529 ML\Prog1 Decision Trees\Wine\my_wine-kaggle.csv', index_label='id')

#Plots
# 1) Alpha vs Accuracy
clf = DecisionTreeClassifier(random_state=0)
path = clf.cost_complexity_pruning_path(X_train, Y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
clfs = []
ccp_alphas = ccp_alphas[:-1]
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, Y_train)
    clfs.append(clf)
print("Number of nodes in the last tree is:{} with cpp_alpha: {}".format(
    clfs[-1].tree_.node_count, ccp_alphas[-1]
))

train_scores = [clf.score(X_train, Y_train) for clf in clfs]
test_scores = [clf.score(X_test, Y_test) for clf in clfs]
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for testing and training")
ax.plot(ccp_alphas, train_scores, marker='o', label ="train", drawstyle = "steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label ="test", drawstyle = "steps-post")
ax.legend()
#the last tree had only one node so removing that
clfs = clfs[:-1]

# 2) Max depth vs Accuracy
max_depth_range = list(range(1, 100))
clfs=[]
for depth in max_depth_range:
    clf = DecisionTreeClassifier(max_depth=depth,
                                 random_state=0)
    clf.fit(X_train, Y_train)
    clfs.append(clf)

train_scores = [clf.score(X_train, Y_train) for clf in clfs]
test_scores = [clf.score(X_test, Y_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("Accuracy")
ax.set_ylabel("Depth")
ax.set_title("Max Depth vs Accuracy")
ax.plot(train_scores, max_depth_range, marker='o', label ="train", drawstyle = "steps-post")
ax.plot(test_scores, max_depth_range, marker='o', label ="test", drawstyle = "steps-post")
ax.legend()

# 3) Max leaf nodes vs Accuracy
max_nodes_range = list(range(2, 1000))
clfs=[]
for nodes in max_nodes_range:
    clf = DecisionTreeClassifier(max_leaf_nodes=nodes,
                                 random_state=0)
    clf.fit(X_train, Y_train)
    clfs.append(clf)

train_scores = [clf.score(X_train, Y_train) for clf in clfs]
test_scores = [clf.score(X_test, Y_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("Accuracy")
ax.set_ylabel("Max leaf Nodes")
ax.set_title("Max leaf Nodes vs Accuracy")
ax.plot(train_scores, max_nodes_range, marker='o', label ="train", drawstyle = "steps-post")
ax.plot(test_scores, max_nodes_range, marker='o', label ="test", drawstyle = "steps-post")
ax.legend()

# 4) Min samples split vs Accuracy
min_samples_split_range = list(range(2, 1000))
clfs=[]
for splits in min_samples_split_range:
    clf = DecisionTreeClassifier(min_samples_split=splits,
                                 random_state=0)
    clf.fit(X_train, Y_train)
    clfs.append(clf)

train_scores = [clf.score(X_train, Y_train) for clf in clfs]
test_scores = [clf.score(X_test, Y_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("Accuracy")
ax.set_ylabel("Min samples split")
ax.set_title("Min samples split vs Accuracy")
ax.plot(train_scores, min_samples_split_range, marker='o', label ="train", drawstyle = "steps-post")
ax.plot(test_scores, min_samples_split_range, marker='o', label ="test", drawstyle = "steps-post")
ax.legend()

# 5) Min impurity deacrease vs Accuracy
clfs=[]
min_impurity_decease_range = list(np.arange(0.0, 30.0, 1.0))
for idecrease in min_impurity_decease_range:
    clf = DecisionTreeClassifier(min_impurity_decrease=idecrease,
                                 random_state=0)
    clf.fit(X_train, Y_train)
    clfs.append(clf)

train_scores = [clf.score(X_train, Y_train) for clf in clfs]
test_scores = [clf.score(X_test, Y_test) for clf in clfs]
fig, ax = plt.subplots()
ax.set_xlabel("Accuracy")
ax.set_ylabel("Min Impurity Decrease")
ax.set_title("Min Impurity Decrease vs Accuracy")
ax.plot(train_scores, min_impurity_decease_range, marker='o', label ="train", drawstyle = "steps-post")
ax.plot(test_scores, min_impurity_decease_range, marker='o', label ="test", drawstyle = "steps-post")
ax.legend()

# 6) Min samples leaf vs Accuracy
clfs=[]
min_samples_leaf_range = list(range(1, 100))
for isample in min_samples_leaf_range:
    clf = DecisionTreeClassifier(min_samples_leaf=isample,
                                 random_state=0)
    clf.fit(X_train, Y_train)
    clfs.append(clf)

train_scores = [clf.score(X_train, Y_train) for clf in clfs]
test_scores = [clf.score(X_test, Y_test) for clf in clfs]
fig, ax = plt.subplots()
ax.set_xlabel("Accuracy")
ax.set_ylabel("Min Samples Leaf")
ax.set_title("Min Samples Leaf vs Accuracy")
ax.plot(train_scores, min_samples_leaf_range, marker='o', label ="train", drawstyle = "steps-post")
ax.plot(test_scores, min_samples_leaf_range, marker='o', label ="test", drawstyle = "steps-post")
ax.legend()
plt.show()


'''
#BOX PLOTS
data = wine_training.iloc[:, 1]
fig = plt.figure(figsize=(10, 7))
plt.boxplot(data)
plt.title("Fixed Acidity")

data = wine_training.iloc[:, 2]
fig = plt.figure(figsize=(10, 7))
plt.boxplot(data)
plt.title("Volatile Acidity")

data = wine_training.iloc[:, 3]
fig = plt.figure(figsize=(10, 7))
plt.boxplot(data)
plt.title("Citric Acid")

data = wine_training.iloc[:, 4]
fig = plt.figure(figsize=(10, 7))
plt.boxplot(data)
plt.title("Residual Sugar")

data = wine_training.iloc[:, 5]
fig = plt.figure(figsize=(10, 7))
plt.boxplot(data)
plt.title("Chlorides")

data = wine_training.iloc[:, 6]
fig = plt.figure(figsize=(10, 7))
plt.boxplot(data)
plt.title("Free Sulfur Dioxide")

data = wine_training.iloc[:, 7]
fig = plt.figure(figsize=(10, 7))
plt.boxplot(data)
plt.title("Total Sulfur Dioxide")

data = wine_training.iloc[:, 8]
fig = plt.figure(figsize=(10, 7))
plt.boxplot(data)
plt.title("Density")

data = wine_training.iloc[:, 9]
fig = plt.figure(figsize=(10, 7))
plt.boxplot(data)
plt.title("pH")

data = wine_training.iloc[:, 10]
fig = plt.figure(figsize=(10, 7))
plt.boxplot(data)
plt.title("Sulphates")

data = wine_training.iloc[:, 11]
fig = plt.figure(figsize=(10, 7))
plt.boxplot(data)
plt.title("Alcohol")
plt.show()
'''
