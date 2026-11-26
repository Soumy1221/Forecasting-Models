import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load data
df = pd.read_csv("india_internet_traffic_synthetic.csv")

# Endogenous variable
y = df["annual_traffic_eb"]

# Exogenous variables
X = df[["price_inr_gb", "jio_dummy"]]

# Fit SARIMAX model
model = SARIMAX(y, exog=X, order=(1,1,1))
results = model.fit()

print(results.summary())

# Forecast next 11 years (2025–2035)
future_X = pd.DataFrame({
    "price_inr_gb": [9]*11,
    "jio_dummy": [1]*11
})

forecast = results.forecast(steps=11, exog=future_X)
print("\nForecast (EB):")
print(forecast)
