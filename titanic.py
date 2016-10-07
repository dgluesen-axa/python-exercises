# Imports
import pandas as pd
import numpy as np
from sklearn import tree


# Load the train and test datasets to create two DataFrames
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# Dimensions and statistics of the data
train.shape
train.describe()
test.shape
test.describe()


# Marking childhood
train["Child"] = 0
train["Child"][train["Age"] < 18] = 1


# Imputations and conversions
train["Age"] = train["Age"].fillna(train["Age"].median())

train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

train["Embarked"] = train["Embarked"].fillna("S")
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2



# Decision trees

# Create the target and features numpy arrays
target = train["Survived"].values
features1 = train[["Pclass", "Sex", "Age", "Fare"]].values

# Fit first decision tree
my_tree1 = tree.DecisionTreeClassifier()
my_tree1 = my_tree1.fit(features1, target)

# Importance and score of the included features
print(my_tree1.feature_importances_)
print(my_tree1.score(features1, target))


# Test of decision tree fit
# Imputations and conversions
test.Fare[152] = test["Fare"].median()
test["Age"] = test["Age"].fillna(test["Age"].median())
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
test["Embarked"] = test["Embarked"].fillna("S")
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2

# Extract the features from the test set
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

# Make prediction using the test set
my_prediction = my_tree1.predict(test_features)
print(my_prediction)

# Create data frame with two columns. Survived contains your predictions
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)

# Write solution to csv file
my_solution.to_csv("my_solution1.csv", index_label = ["PassengerId"])


# Create a new array with features added
test_features2 = test[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values
features2 = train[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values


# Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5
max_depth = 10
min_samples_split = 5
my_tree2 = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split, random_state = 1)
my_tree2 = my_tree2.fit(features2, target)


# Print the score of the new decison tree and output
print(my_tree2.score(features2, target))
my_prediction2 = my_tree2.predict(test_features2)
my_solution2 = pd.DataFrame(my_prediction2, PassengerId, columns = ["Survived"])
my_solution2.to_csv("my_solution2.csv", index_label = ["PassengerId"])


# Create train2 with the newly defined feature
train2 = train.copy()
train2["family_size"] = train2["SibSp"] + train2["Parch"] + 1
test2 = test.copy()
test2["family_size"] = test2["SibSp"] + test2["Parch"] + 1

# Create a new feature set and add the new feature
test_features3 = test2[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked", "family_size"]].values
features3 = train2[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked", "family_size"]].values

# Define the tree classifier, then fit the model
my_tree3 = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split, random_state = 1)
my_tree3 = my_tree3.fit(features3, target)

# Print the score of the new decison tree and output
print(my_tree3.score(features3, target))
my_prediction3 = my_tree3.predict(test_features3)
my_solution3 = pd.DataFrame(my_prediction3, PassengerId, columns = ["Survived"])
my_solution3.to_csv("my_solution3.csv", index_label = ["PassengerId"])



# Random forests

# Additional import
from sklearn.ensemble import RandomForestClassifier

# Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_forest, target)

# Print the score of the fitted random forest
print(my_forest.score(features_forest, target))

# Compute predictions on our test set features then print the length of the prediction vector and output
test_features_forest = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
pred_forest = my_forest.predict(test_features_forest)
print(len(pred_forest))
my_solution_forest = pd.DataFrame(pred_forest, PassengerId, columns = ["Survived"])
my_solution_forest.to_csv("my_solution_forest.csv", index_label = ["PassengerId"])









