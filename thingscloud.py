import requests

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

    # --- Approximate derived model (non-medical) ---
    systolic = 0.45 * heart_rate + 0.5 * spo2 - 0.2 * temperature + 40
    diastolic = 0.35 * heart_rate + 0.3 * spo2 - 0.15 * temperature + 20

    # rounding
    systolic = round(systolic, 1)
    diastolic = round(diastolic, 1)
    return systolic, diastolic


response = requests.get(url)
if response.status_code == 200:
    data = response.json()
    humidity=float(data.get("field4", 0))
    temp = float(data.get("field3", 0))  # Temperature °C
    heart_rate = float(data.get("field1", 0))  # Heart rate BPM
    spo2 = float(data.get("field2", 0))  # SpO2 %

    print(f"Temperature: {temp} °C")
    print(f"Heart Rate: {heart_rate} bpm")
    print(f"SpO2: {spo2} %")

    systolic, diastolic = estimate_bp(temp, heart_rate, spo2)
    print("\n--- Derived Blood Pressure ---")
    print(f"Systolic BP (mmHg): {systolic}")
    print(f"Diastolic BP (mmHg): {diastolic}")

else:
    print(f"Error fetching data: {response.status_code}")
