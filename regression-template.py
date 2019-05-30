#Regression template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # ensure that independent variable is a matrix and not a array.
y = dataset.iloc[:, 2].values
'''
#No training set and test set splitting because there is not enough data available for the split
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
'''
# Feature Scaling (Most of the libraries we use, we dont need to do fearure scaling manually ; the library itself takes care of it)
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Regression model to the dataset
 #Create your regressor here

#Predicting a new result
y_pred = regressor.predict(6.5)

#visualising the polynomial regression results
plt.scatter(X,y,color="red")
plt.plot(X, regressor.predict(X),color="blue")  
plt.title('Truth or bluff (Regression model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


#visualising the polynomial regression results (for higher resolution and smoother curves)
X_grid = np.arrange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color="red")
plt.plot(X, regressor.predict(X_grid),color="blue")  
plt.title('Truth or bluff (Regression model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
