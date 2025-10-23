import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


#column names: Index(['Player', 'Age', 'Team', 'Pos', 'G', 'GS', 'Tgt', 'Rec', 'Yds', 'Y/R','TD', '1D', 'Succ%', 'Lng', 'R/G', 'Y/G', 'Ctch%', 'Y/Tgt', 'Fmb']
raw_data = pd.read_csv('receivingstats.csv', index_col='Rk')
data = raw_data.copy()
columnstodrop = ["Player", "Age", "Team", "Pos"]
inputdata = data.drop(columns=columnstodrop, axis=1)
cleaninput = inputdata.dropna()


x_train, x_test, y_train, y_test = train_test_split(
    cleaninput.drop(['Rec'], axis=1), cleaninput["Rec"], test_size=0.4
)
param_grid = {
    'max_iter': [100, 300, 500],
    'tol': [1e-3, 1e-4],
    'alpha_1': [1e-6, 1e-5],
    'lambda_1': [1e-6, 1e-5],
}

grid = GridSearchCV(BayesianRidge(), param_grid, cv=5, verbose=1)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)

print("Best Parameters:", grid.best_params_)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("R-squared (R²):", r2_score(y_test, y_pred))

plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Bayesian Ridge Regression Predictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # identity line
plt.show()
