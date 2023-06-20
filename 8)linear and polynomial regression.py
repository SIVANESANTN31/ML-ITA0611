import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv("C:/Users/Asus/OneDrive/Documents/ML/DATASET/Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
"""
Training the Linear Regression model on the Whole dataset
A Linear regression algorithm is used to create a model.
 A LinearRegression function is imported from sklearn.linear_model library.
 """

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Linear Regression classifier model
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
from sklearn.linear_model import LinearRegression
reg = LinearRegression(normalize_X=True)

"""
Training the Polynomial Regression model on the Whole dataset
A polynomial regression algorithm is used to create a model.
"""
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
#Polynomial Regression classifier model
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
"""
Visualising the Linear Regression results
Here scatter plot is used to visualize the results. The title of the plot is set to Truth or Bluff 
(Linear Regression), xlabel is set to Position Level , and ylabel is set to Salary.
"""
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
#Visualising the Polynomial Regression results
"""
The title of the plot is set to Truth or Bluff (Polynomial Regression), xlabel is set to Position level,
 and ylabel is set to Salary.
"""
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
