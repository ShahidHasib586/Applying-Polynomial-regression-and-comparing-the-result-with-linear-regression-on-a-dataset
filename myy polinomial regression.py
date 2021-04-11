#importing the libraries

import numpy as np
import pandas as pd
#import matplotlib as plt
from matplotlib import pyplot as plt
#reading the dataset

dataset = pd.read_csv('Position_Salaries.csv')

#putting indepandent variabls value

X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

#Linear Regression

from sklearn.linear_model import LinearRegression
 
lin_reg = LinearRegression()

lin_reg.fit(X, Y)

#polynomial regression

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

#visualizing linear regression

plt.scatter(X, Y, color ='red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title ('Truth or Bluff(linear Regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

#visualizing polynomial regression
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y, color ='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title ('Truth or Bluff(linear Regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

#predict new result with linear regression
lin_reg.predict([[6.5]])
#predict new salary with polynomial regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
