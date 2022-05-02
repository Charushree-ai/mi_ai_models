# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# import data
fueleconomy_df = pd.read_csv('FuelEconomy.csv')

# create testing and training dataset
# save X in dataframe
X = fueleconomy_df[['Horse Power']]
# save y in series
y = fueleconomy_df['Fuel Economy (MPG)']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# train the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(fit_intercept=True)
regressor.fit(X_train, y_train)

print('Linear Model Coeff (m):', regressor.coef_)
print('Linear Model Coeff (b):', regressor.intercept_)

# test the model
y_predict = regressor.predict(X_test)

plt.scatter(X_train, y_train, color = 'gray')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.xlabel('Horse Power (HP)')
plt.ylabel('MPG')
plt.title('HP vs. MPG (Training Set)')
plt.show()

plt.scatter(X_test, y_test, color = 'gray')
plt.plot(X_test, regressor.predict(X_test), color = 'blue')
plt.xlabel('Horse Power (HP)')
plt.ylabel('MPG')
plt.title('HP vs. MPG (Testing Set)')
plt.show()

HP = 500
HP = np.asarray(HP)
HP = np.reshape(HP,(-1,1))

MPG = regressor.predict(HP)
MPG

print(HP, MPG)