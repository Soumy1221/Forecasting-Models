import pandas as pd
from prophet import Prophet

# Load data
df = pd.read_csv("india_internet_traffic_synthetic.csv")

# Prophet format
df_prophet = df.rename(columns={"year": "ds", "annual_traffic_eb": "y"})
df_prophet["ds"] = pd.to_datetime(df_prophet["ds"], format="%Y")

model = Prophet()
model.fit(df_prophet)

future = model.make_future_dataframe(periods=11, freq="Y")
forecast = model.predict(future)

print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(11))
