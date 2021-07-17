import pandas as pd
import dvc.api
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn import tree
import pickle
import json

# Getting data
repo = "https://github.com/SohamIITBombay/MLOps_Assignment.git"
path = "data/creditcard.csv"

with dvc.api.open(repo=repo, path=path, mode="r") as fd:
    df = pd.read_csv(fd)



# Wrangling
df.drop_duplicates(inplace=True)



# Train test split
data = df.values
train_data, test_data = train_test_split(data, test_size=0.2, random_state=100)

X_train = train_data[:, :-1]
Y_train = train_data[:, -1]

X_test = test_data[:, :-1]
Y_test = test_data[:, -1]



# Hyperparameters
max_depth = 5
min_leaf = 10


# Training
classifier_DT = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, min_samples_leaf=min_leaf)

classifier_DT.fit(X_train, Y_train)

Y_pred = classifier_DT.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred) * 100
f1_score = f1_score(Y_test, Y_pred)

print("Accuracy: ", accuracy)
print("\n")
print("F1 Score: ", f1_score)



# Saving files

pd.DataFrame(train_data).to_csv("C:\\Users\\soham\\Desktop\\MLOps\\MLOps_Assignment\\data\\prepared\\train.csv")
pd.DataFrame(test_data).to_csv("C:\\Users\\soham\\Desktop\\MLOps\\MLOps_Assignment\\data\\prepared\\test.csv")

pickle.dump(classifier_DT, open("C:\\Users\\soham\\Desktop\\MLOps\\MLOps_Assignment\\models\\model.pkl", "wb"))


perfMetrics = {}
perfMetrics["Accuracy"] = accuracy
perfMetrics["F1 Score"] = f1_score

with open("C:\\Users\\soham\\Desktop\\MLOps\\MLOps_Assignment\\metrics\\acc_f1.json", "w") as outfile:
    json.dump(perfMetrics, outfile)




