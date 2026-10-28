import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("city_day.csv")

# Features and target
features = ["PM2.5", "PM10", "NO2", "CO"]
target = "AQI"

# Convert columns to numeric
for col in features + [target]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows without a known AQI target
df = df.dropna(subset=[target])
df = df[df[target] >= 0]

X = df[features]
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Impute missing pollutant readings using training data
model = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
)

# Train
print("Training model...")
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print("\nModel evaluation")
print("MAE:", round(mean_absolute_error(y_test, predictions), 2))
print("R2 score:", round(r2_score(y_test, predictions), 3))

# Save model and feature names
joblib.dump(
    {"model": model, "features": features},
    "aqi_model.pkl"
)

print("\nModel saved as aqi_model.pkl")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
