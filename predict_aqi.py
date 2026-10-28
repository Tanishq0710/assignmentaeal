import joblib
import pandas as pd

# Load trained model
saved = joblib.load("aqi_model.pkl")
model = saved["model"]
features = saved["features"]

# Simulated roadside sensor readings
sensor_readings = {
    "PM2.5": 65.0,
    "PM10": 110.0,
    "NO2": 30.0,
    "CO": 1.2
}

# Prepare input in the correct feature order
input_data = pd.DataFrame([sensor_readings])[features]

# Predict AQI locally
predicted_aqi = float(model.predict(input_data)[0])

print("\n===== SMART CITY AQI MONITOR =====")
print("Sensor readings:")
for name, value in sensor_readings.items():
    print(f"{name}: {value}")

print(f"\nPredicted AQI: {predicted_aqi:.2f}")
print("Inference mode: LOCAL")
print("Cloud dependency: NONE")
