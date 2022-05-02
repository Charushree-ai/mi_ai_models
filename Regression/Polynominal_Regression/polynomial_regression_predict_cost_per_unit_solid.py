# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Import dataset
economies_cost = pd.read_csv("EconomiesOfScale.csv")

X = economies_cost[["Number of Units"]]
y = economies_cost["Manufacturing Cost"]

# Note that we used the entire dataset for training only
X_train = X
y_train = y

# train the model using 5 degree
poly_regressor = PolynomialFeatures(degree=5)
X_columns = poly_regressor.fit_transform(X_train)
regressor = LinearRegression()
regressor.fit(X_columns, y_train)

# result
print('Linear Model Coefficient (m): ', regressor.coef_)
print('Linear Model Coefficient (b): ', regressor.intercept_)

# prediction
y_predict = regressor.predict(poly_regressor.fit_transform(X_train))

# visualize the results
plot.scatter(X_train, y_train, color = 'red')
plot.plot(X_train, y_predict, color = 'blue')
plot.ylabel('Cost Per Unit Sold [dollars]')
plot.xlabel('Number of Units [in Millions]')
plot.title('Unit Cost vs. Number of Units [in Millions](Training dataset)')
plot.show()