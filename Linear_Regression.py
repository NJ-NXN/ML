from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

x = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([40000, 45000, 50000, 55000, 60000, 65000])

x_train, y_train, x_test, y_test = x[:4], y[:4], x[4:], y[4:]

model = LinearRegression()
model.fit(x_train, y_train)

predictions = model.predict(x_test)

mae = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Absolute Error:", mae)
print("R^2 Score:", r2)

print("Slope:", model.coef_[0])
print("R^2 Score:", r2)

future_salary = model.predict([[7]])
print("Predicted salary for 7 years of experience:", future_salary[0])