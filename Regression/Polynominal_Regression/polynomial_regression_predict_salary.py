# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns

# Import data
employee_salary = pd.read_csv("Employee_Salary.csv")

# Create train and test data
X = employee_salary[["Years of Experience"]]
Y = employee_salary["Salary"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Train model
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree=5)
X_columns = poly_regressor.fit_transform(X_train)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_columns, y_train)

# prediction result of train model
y_predict = regressor.predict(poly_regressor.fit_transform(X_train))

# Visualize train model
plot.scatter(X_train, y_train, color = 'red')
plot.plot(X_train, y_predict, color = 'blue')
plot.ylabel('Salary/Year [dollars]')
plot.xlabel('Years of Experience')
plot.title('Salary vs. Years of Experience (Training dataset)')
# plot.show()

plot.scatter(X_test, y_test, color = 'red')
plot.plot(X_test, regressor.predict(poly_regressor.fit_transform(X_test)), color = 'blue')
plot.ylabel('Salary/Year [dollars]')
plot.xlabel('Years of Experience')
plot.title('Salary vs. Years of Experience (Testing dataset)')
plot.show()