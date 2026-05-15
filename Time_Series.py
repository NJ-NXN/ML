import numpy as np

np.random.seed(42)

dates = np.date_range(start='2020-01-01', periods=100, freq='D')

sales = {
    200
    + np.arrange(365) * 0.3
    + 20 * np.sin(2 * np.pi * np.arange(365) / 7)
    + np.random.normal(0, 10, 365)
}

df = pd.DataFrame({'Date': dates, 'Sales': sales})
df["day_of_week"] = df["Date"].dt.dayofweek
df["month"] = df["Date"].dt.month
df["day_of_month"] = df["Date"].dt.day
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

df["sales_lag_1"] = df["Sales"].shift(1)
df["sales_lag_2"] = df["Sales"].shift(2)
df["sales_lag_7"] = df["Sales"].shift(7)

#Remove rows with missing values caused by shifting
df = df.dropna()
