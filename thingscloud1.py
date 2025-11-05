import requests
from datetime import datetime, timezone, timedelta

# ThingSpeak channel details
CHANNEL_ID = "3134881"
READ_API_KEY = "0VE22L3KY6VLEO3O"

url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds/last.json?api_key={READ_API_KEY}"

def estimate_bp(temperature, heart_rate, spo2):
    """
    Estimate systolic and diastolic blood pressure 
    from temperature (°C), heart rate (bpm), and SpO2 (%)
    using heuristic model.
    """
    systolic = 0.45 * heart_rate + 0.5 * spo2 - 0.2 * temperature + 40
    diastolic = 0.35 * heart_rate + 0.3 * spo2 - 0.15 * temperature + 20
    return round(systolic, 1), round(diastolic, 1)


response = requests.get(url)
if response.status_code == 200:
    data = response.json()

    # Extract fields
    humidity=float(data.get("field4", 0))
    temp = float(data.get("field3", 0))  # Temperature °C
    heart_rate = float(data.get("field1", 0))  # Heart rate BPM
    spo2 = float(data.get("field2", 0))  # SpO2 %
    timestamp_str = data.get("created_at")  # Example: "2025-10-28T03:10:22Z"

    # Convert ThingSpeak UTC timestamp to datetime
    data_time = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    now_time = datetime.now(timezone.utc)

    # Check if updated within last 2 minutes
    time_diff = now_time - data_time
    minutes_diff = time_diff.total_seconds() / 60

    print(f"Data Timestamp (UTC): {data_time}")
    print(f"Current Time (UTC):   {now_time}")
    print(f"Data Age: {minutes_diff:.2f} minutes")

    if minutes_diff <= 2:
        systolic, diastolic = estimate_bp(temp, heart_rate, spo2)
        print("\n--- Latest Sensor Data ---")
        print(f"Temperature: {temp} °C")
        print(f"Heart Rate: {heart_rate} bpm")
        print(f"SpO2: {spo2} %")

        print("\n--- Derived Blood Pressure ---")
        print(f"Systolic BP (mmHg): {systolic}")
        print(f"Diastolic BP (mmHg): {diastolic}")
    else:
        print("\n⚠️ Data not updated within the last 2 minutes. Skipping BP calculation.")
else:
    print(f"Error fetching data: {response.status_code}")
