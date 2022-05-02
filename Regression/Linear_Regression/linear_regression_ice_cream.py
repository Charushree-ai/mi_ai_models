# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

# Import data
IceCream = pd.read_csv("IceCreamData.csv")

print(IceCream.head())

# create testing and training dataset
# save y in series
y = IceCream['Revenue']
# save x in dataframe
X = IceCream[['Temperature']]

# Create train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# train the model
regressor = LinearRegression(fit_intercept=True)
regressor.fit(X_train, y_train)

print('Linear Model Coefficient (m): ', regressor.coef_)
print('Linear Model Coefficient (b): ', regressor.intercept_)

# test the model (output will be in numpy array format)
y_predict = regressor.predict(X_test)

plt.scatter(X_train, y_train, color='gray')
plt.plot(X_train, regressor.predict(X_train), color='red')
plt.ylabel('Revenue [dollars]')
plt.xlabel('Temperature [degC]')
plt.title('Revenue Generated vs. Temperature @Ice Cream Stand(Training dataset)')
# plt.show()

# VISUALIZE TEST SET RESULTS
plt.scatter(X_test, y_test, color='gray')
plt.plot(X_test, regressor.predict(X_test), color='red')
plt.ylabel('Revenue [dollars]')
plt.xlabel('Hours')
plt.title('Revenue Generated vs. Hours @Ice Cream Stand(Test dataset)')
plt.show()

T = [35]
T = np.asarray(T)
T = np.reshape(T, (-1, 1))

y_predict = regressor.predict(T)
print(y_predict)
