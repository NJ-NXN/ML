from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

temperature = np.array([60, 65, 70, 75, 80, 85, 90]).reshape(-1, 1)
sales = np.array([125, 118, 147, 150, 183, 193, 237])

x_train, x_test, y_train, y_test = train_test_split(temperature, sales, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

predictions = model.predict(x_test)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

new_temp = np.array([[72]])
predicted_sales = model.predict(new_temp)
print("Predicted sales for 72 degrees:", predicted_sales[0])

plt.scatter(temperature, sales, color='blue', label='Actual Sales')
plt.plot(temperature, model.predict(temperature), color='red', label='Regression Line')
plt.scatter(new_temp, predicted_sales, color='green', marker='X', s=100, label=f'Prediction at 72° ({predicted_sales[0]:.1f})')
plt.xlabel('Temperature (°F)')
plt.ylabel('Sales')
plt.title('Sales vs Temperature')
plt.legend()
plt.show()

