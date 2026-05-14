import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

ticker = "NVDA"
start_date = "2020-01-01"
end_date = "2026-05-14"
forecast_days = 252
num_simulations = 10000

# Fetch data
data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

# Flatten array to prevent multi-index issues in newer yfinance versions
prices = data["Close"].dropna().to_numpy().flatten()

# Calculate daily log returns
log_returns = np.log(prices[1:] / prices[:-1])

S0 = prices[-1]
mu = log_returns.mean()
sigma = log_returns.std()

dt = 1 
paths = np.zeros((forecast_days + 1, num_simulations))
paths[0] = S0


for t in range(1, forecast_days + 1):
    Z = np.random.standard_normal(num_simulations)
    paths[t] = paths[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(paths[:, :100], color='blue', alpha=0.1)
average_path = paths.mean(axis=1)
plt.plot(average_path, color='red', linewidth=2, label='Average Path')
plt.xlabel('Trading Days')
plt.ylabel('Stock Price')
plt.title(f'Monte Carlo Simulation for {ticker} (100 Sample Paths)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()