# Implementation-of-Logistic-Regression-Using-Gradient-Descent
## DATE: 3/2/2026
## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start the program.
2. Data preprocessing: Cleanse data,handle missing values,encode categorical variables.
3. Model Training: Fit logistic regression model on preprocessed data.
4.Model Evaluation: Assess model performance using metrics like accuracyprecisioon,recall.
5.Prediction: Predict placement status for new student data using trained model.
6.End the program
7.Program:  

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: K.AshwinNehrej
RegisterNumber:  25015594
*/
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Placement_Data.csv")

data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})

X = data[['ssc_p', 'mba_p']].values
y = data['status'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

m = len(y)
X = np.c_[np.ones(m), X]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, theta):
    h = sigmoid(X @ theta)
    return (-1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))

theta = np.zeros(X.shape[1])
alpha = 0.1
cost_history = []

for i in range(500):
    z = X @ theta
    h = sigmoid(z)
    gradient = (1/m) * X.T @ (h - y)
    theta = theta - alpha * gradient
    
    cost = cost_function(X, y, theta)
    cost_history.append(cost)

y_pred = (sigmoid(X @ theta) >= 0.5).astype(int)

accuracy = np.mean(y_pred == y) * 100
print("Weights:", theta)
print("Accuracy:", accuracy, "%")

plt.figure()
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Logistic Regression using Gradient Descent")
plt.show()
```

## Output:
```
Weights: [ 1.26712229  2.20688701 -0.59115221]
Accuracy: 82.32558139534883 %
```


<img width="1034" height="693" alt="Screenshot 2026-02-03 092058" src="https://github.com/user-attachments/assets/b94cc84d-f0a2-4846-a7e9-5b0da8023a4d" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

