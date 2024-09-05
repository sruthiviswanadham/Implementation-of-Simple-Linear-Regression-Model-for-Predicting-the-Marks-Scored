# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the   graph.
5. Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:

/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: viswanadham venkata sai sruthi
RegisterNumber: 212223100061 
*/

```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
```
```
dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
```
![OUTPUT](image.png)
```
dataset.info()
```
![OUTPUT](image-1.png)
```
#assigning hours to X & scores to Y
X = dataset.iloc[:,:-1].values
print(X)
Y = dataset.iloc[:,-1].values
print(Y)
```
![OUTPUT](image-2.png)
```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
```
```
X_train.shape,X_test.shape
```
![OUTPUT](image-3.png)
```
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,Y_train)
```
![OUTPUT](image-4.png)
```
Y_pred = reg.predict(X_test)
print(Y_pred)
print(Y_test)
```
![OUTPUT](image-5.png)
```
plt.scatter(X_train,Y_train,color="green")
plt.plot(X_train,reg.predict(X_train),color="red")
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,reg.predict(X_test),color="silver")
plt.title('Test set(H vs S)') 
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()         
```
![OUTPUT](image-6.png)
![OUTPUT](image-7.png)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
