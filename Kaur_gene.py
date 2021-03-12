from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier


#READING TRAINING DATA FROM GENE TRAIN CSV FILE
gene_training = pd.read_csv(r'C:\Users\Vinni\Desktop\Sem 3 Spring 2021\CS 529 ML\Prog1 Decision Trees\Gene\gene_1\gene-train.csv',header=None)
X_train = gene_training.iloc[:, 1:61]
Y_train = gene_training.iloc[:, -1:]
#READING TESTING DATA FROM GENE TEST CSV FILE index starts from 1 because the 0th column is index
gene_testing = pd.read_csv(r'C:\Users\Vinni\Desktop\Sem 3 Spring 2021\CS 529 ML\Prog1 Decision Trees\Gene\gene_1\gene-test.csv', header=None)
X_test = gene_testing.iloc[:, 1:61]
Y_test = gene_testing.iloc[:, -1:]
#Decision Tree with default parameters
tree = DecisionTreeClassifier(random_state=1)
classifier = tree.fit(X_train, Y_train)
print("Decision Tree with default parameters")
print('Accuracy for training data: %.3f'%tree.score(X_train, Y_train))
print('Accuracy for testing data: %.3f'%tree.score(X_test, Y_test))

#Decision Tree with changed parameters
tree = DecisionTreeClassifier(random_state=5, ccp_alpha=0.0014, max_depth=9, min_impurity_decrease=0, max_leaf_nodes=20, min_samples_leaf=1, min_samples_split=50)
classifier = tree.fit(X_train, Y_train)
print("Decision Tree with modified parameters")
print('Accuracy for training data: %.3f'%tree.score(X_train, Y_train))
print('Accuracy for testing data: %.3f'%tree.score(X_test, Y_test))

#Random Forest with default parameters
y=np.ravel(Y_train)
clf = RandomForestClassifier(random_state=1)
clf.fit(X_train, y)
print("Random Forest with default parameters")
print('Accuracy for training data: %.3f'%clf.score(X_train, Y_train))
print('Accuracy for testing data: %.3f'%clf.score(X_test, Y_test))

#Random Forest with changed parameters
y=np.ravel(Y_train)
clf = RandomForestClassifier(criterion='gini', max_features='log2', random_state=10, max_depth=15, n_estimators=1000)
clf.fit(X_train, y)
print("Random Forest with modified parameters")
print('Accuracy for training data: %.3f'%clf.score(X_train, Y_train))
print('Accuracy for testing data: %.3f'%clf.score(X_test, Y_test))


#GENE KAGGLE
gene_kaggle= pd.read_csv(r'C:\Users\Vinni\Desktop\Sem 3 Spring 2021\CS 529 ML\Prog1 Decision Trees\Gene\gene_1\gene-kaggle.csv', header=None)
X_kaggle = gene_kaggle.iloc[:, 1:61]
ktree =RandomForestClassifier(criterion='gini', max_features='log2', random_state=10, max_depth=15, n_estimators=1000)
#96.33 accuracy with ktree =RandomForestClassifier(criterion='gini', max_features='log2',random_state=10, max_depth=15,n_estimators=1000)
y=np.ravel(Y_train)
ktree.fit(X_train, y)
kaggle_prediction = ktree.predict(X_kaggle)
#forming a dataframe with columns id and class to save the predictions into a csv file
frame = pd.DataFrame(kaggle_prediction, columns=['class'])
frame.index = frame.index+1
frame.to_csv(r'C:\Users\Vinni\Desktop\Sem 3 Spring 2021\CS 529 ML\Prog1 Decision Trees\Gene\gene_1\my_gene_kaggle.csv', index_label='id')

#PLOTS
#1) ALPHA VALUE AND ALPHA VS ACCURACY PLOT
clf = DecisionTreeClassifier(random_state=20)
path = clf.cost_complexity_pruning_path(X_train, Y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
clfs = []
ccp_alphas = ccp_alphas[:-1]
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=20, ccp_alpha=ccp_alpha)
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

clfs = clfs[:-1]
#2) MAX DEPTH VS ACCURACY
max_depth_range = list(range(1, 50))
clfs=[]
for depth in max_depth_range:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=20)
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

#3) MAX LEAF NODES VS ACCURACY
max_nodes_range = list(range(2, 150))
clfs=[]
for nodes in max_nodes_range:
    clf = DecisionTreeClassifier(max_leaf_nodes=nodes, random_state=20)
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

#4) MIN SAMPLES SPLIT VS ACCURACY
min_samples_split_range = list(range(2, 900))
clfs=[]
for splits in min_samples_split_range:
    clf = DecisionTreeClassifier(min_samples_split=splits, random_state=20)
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

#5) MIN IMPURITY DECREASE VS ACCURACY
clfs=[]
min_impurity_decease_range = list(np.arange(0.0, 30.0, 1.0))

for idecrease in min_impurity_decease_range:
    clf = DecisionTreeClassifier(min_impurity_decrease=idecrease, random_state=20)
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

#6) RANDOM STATE VS ACCURACY
clfs=[]
random_state_range = list(range(0, 50))

for rsrange in random_state_range:
    clf = DecisionTreeClassifier(random_state=rsrange)
    clf.fit(X_train, Y_train)
    clfs.append(clf)

train_scores = [clf.score(X_train, Y_train) for clf in clfs]
test_scores = [clf.score(X_test, Y_test) for clf in clfs]
fig, ax = plt.subplots()
ax.set_xlabel("Accuracy")
ax.set_ylabel("Random State")
ax.set_title("Random state vs Accuracy")
ax.plot(train_scores, random_state_range, marker='o', label ="train", drawstyle = "steps-post")
ax.plot(test_scores, random_state_range, marker='o', label ="test", drawstyle = "steps-post")
ax.legend()

#7) MIN SAMPLES LEAF VS ACCURACY
clfs=[]
min_samples_leaf_range = list(range(1, 100))

for isample in min_samples_leaf_range:
    clf = DecisionTreeClassifier(min_samples_leaf=isample, random_state=20)
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




