import pandas as pd

df = pd.read_csv("city_day.csv")

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nRequired columns:")
for col in ["PM2.5", "PM10", "NO2", "CO", "Temp", "AQI", "T"]:
    print(f"{col}: {col in df.columns}")
