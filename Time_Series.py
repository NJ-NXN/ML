import numpy as np
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

dates = pd.date_range(start='2020-01-01', periods=365, freq='D')

sales = (
    200
    + np.arange(365) * 0.3
    + 20 * np.sin(2 * np.pi * np.arange(365) / 7)
    + np.random.normal(0, 10, 365)
)

df = pd.DataFrame({'Date': dates, 'Sales': sales})
df["day_of_week"] = df["Date"].dt.dayofweek
df["month"] = df["Date"].dt.month
df["day_of_month"] = df["Date"].dt.day
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
df["sales_lag_1"] = df["Sales"].shift(1)
df["sales_lag_2"] = df["Sales"].shift(2)
df["sales_lag_7"] = df["Sales"].shift(7)

# Remove rows with missing values caused by shifting
df = df.dropna()
features = ["day_of_week", "month", "day_of_month", "is_weekend", "sales_lag_1", "sales_lag_2", "sales_lag_7"]
X = df[features]
y = df["Sales"]

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

dates_test = df["Date"].iloc[split_index:]

model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train) 

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)

plt.figure(figsize=(12, 6))
plt.plot(df["Date"].iloc[:split_index], y_train, color='gray', label='Training Data')
plt.plot(dates_test, y_test, color='blue', label='Actual Sales (Test Data)')
plt.plot(dates_test, predictions, color='red', linestyle='--', label='XGBoost Predictions')

plt.title('XGBoost Time-Series Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()