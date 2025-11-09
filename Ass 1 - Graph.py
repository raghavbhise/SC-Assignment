import numpy as np
import matplotlib.pyplot as plt

# --- Membership Functions (from your Java code) ---

# Temperature MF (-10 to 20 °C)
def fuzzify_temperature(x):
    return np.maximum(0, np.minimum(1, (x + 10) / 30))

# Moisture MF (0 to 100%)
def fuzzify_moisture(x):
    return np.maximum(0, np.minimum(1, x / 100))

# Days MF (0 to 5)
def fuzzify_days(x):
    return np.maximum(0, np.minimum(1, x / 5))


# --- Plotting ---

# Temperature range
temp = np.linspace(-10, 20, 200)
temp_mf = fuzzify_temperature(temp)

# Moisture range
moisture = np.linspace(0, 100, 200)
moisture_mf = fuzzify_moisture(moisture)

# Days range
days = np.linspace(0, 5, 200)
days_mf = fuzzify_days(days)

plt.figure(figsize=(14, 4))

# Plot Temperature MF
plt.subplot(1, 3, 1)
plt.plot(temp, temp_mf, 'b', linewidth=2)
plt.title("Temperature MF (-10°C to 20°C)")
plt.xlabel("Temperature (°C)")
plt.ylabel("Membership Value")
plt.ylim(0, 1.1)
plt.grid(True)

# Plot Moisture MF
plt.subplot(1, 3, 2)
plt.plot(moisture, moisture_mf, 'g', linewidth=2)
plt.title("Moisture MF (0% to 100%)")
plt.xlabel("Moisture (%)")
plt.ylabel("Membership Value")
plt.ylim(0, 1.1)
plt.grid(True)

# Plot Days MF
plt.subplot(1, 3, 3)
plt.plot(days, days_mf, 'r', linewidth=2)
plt.title("Days MF (0 to 5)")
plt.xlabel("Days in Fridge")
plt.ylabel("Membership Value")
plt.ylim(0, 1.1)
plt.grid(True)

plt.tight_layout()
plt.show()