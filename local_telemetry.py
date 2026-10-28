import json
from datetime import datetime, timezone
import joblib
import pandas as pd

saved = joblib.load("aqi_model.pkl")
model = saved["model"]
features = saved["features"]

# Simulated roadside sensor readings
readings = {
    "PM2.5": 65.0,
    "PM10": 110.0,
    "NO2": 30.0,
    "CO": 1.2
}

X = pd.DataFrame([readings])[features]
predicted_aqi = float(model.predict(X)[0])

# IoT-style telemetry message
message = {
    "deviceId": "AQI-Edge-Device",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "pollutants": readings,
    "predictedAQI": round(predicted_aqi, 2),
    "inferenceMode": "local"
}

with open("aqi_telemetry.json", "w") as f:
    json.dump(message, f, indent=4)

print(json.dumps(message, indent=4))
print("\nSaved telemetry to aqi_telemetry.json")
