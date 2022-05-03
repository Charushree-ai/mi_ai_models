# import libraries
import numpy as np
import pandas as pd
import matplotlib as plt
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns",500)

# read data
ecommerce_data = pd.read_csv("Ecommerce Customers")
print(ecommerce_data.head())

# create train and test data
X = ecommerce_data[["Length of Membership", "Time on App", "Time on Website", "Avg. Session Length"]]
y = ecommerce_data["Yearly Amount Spent"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# train the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print("Linear Model coef:",regressor.coef_)
print("Linear Model intercept:",regressor.intercept_)

y_prediction = regressor.predict(X_test)