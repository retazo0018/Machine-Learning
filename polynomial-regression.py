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
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

#visualising the linear regression results
plt.scatter(X,y,color="red")
plt.plot(X, lin_reg.predict(X),color="blue")
plt.title('Truth or bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#visualising the polynomial regression results
plt.scatter(X,y,color="red")
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)),color="blue")  #Don't use X_poly , coz X_poly is aldready defined for other use. (Generalisation)
plt.title('Truth or bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#visualising the polynomial regression results (advancement)
X_grid = np.arange(min(X),max(X),0.1) #creates a array from min(X) to max(X) incremented by 1
X_grid = X_grid.reshape((len(X_grid)),1) # 1st arg - length, 2nd arg - number of columns
plt.scatter(X,y,color="red")
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)),color="blue")  #Don't use X_poly , coz X_poly is aldready defined for other use. (Generalisation)
plt.title('Truth or bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#predicting a new result with Linear regression
lin_reg.predict(6.5)

#Predicting a new result with Polynomial regression
lin_reg2.predict(poly_reg.fit_transform(6.5))


