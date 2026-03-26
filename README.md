# BLENDED_LEARNING
# Implementation of Decision Tree Model for Tumor Classification

## AIM:
To implement and evaluate a Decision Tree model to classify tumors as benign or malignant using a dataset of lab test results.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset, separate features and target, scale the features using MinMaxScaler, and encode the target labels using LabelEncoder.

2.Split the dataset into training and testing sets using train_test_split() with stratified sampling.

3.Train a Logistic Regression model with L2 regularization (multinomial) on the training data and make predictions on the test data.

4.Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrix, then visualize the confusion matrix using a heatmap.

## Program:
```
/*
Program to  implement a Decision Tree model for tumor classification.
Developed by: SHRIHARI M
RegisterNumber: 212225230265
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('tumor.csv')
print(data.head())
print(data.columns)
print(data.head())
print(data.columns)
X = data.drop(columns=['Class']) 
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Reds")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

## Output:
<img width="643" height="737" alt="Screenshot 2026-03-26 103712" src="https://github.com/user-attachments/assets/891981e1-705c-47c5-ba38-e99be641a3cf" />


<img width="708" height="525" alt="image" src="https://github.com/user-attachments/assets/78404526-9452-4add-935d-8ac3952b81df" />




## Result:
Thus, the Decision Tree model was successfully implemented to classify tumors as benign or malignant, and the model’s performance was evaluated.
