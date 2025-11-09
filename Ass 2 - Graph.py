import numpy as np
import matplotlib.pyplot as plt

# --- Membership Functions ---
def fitness_level(x):
    return x / 100.0  # Normalized goal input

def deficiency(x):
    return 1 - (x / 100.0)  # Deficiency = 1 - fitness

# Range for input values (0% to 100%)
x = np.linspace(0, 100, 200)

# Calculate MF values
fitness_mf = fitness_level(x)
deficiency_mf = deficiency(x)

# --- Plot ---
plt.figure(figsize=(7,5))

plt.plot(x, fitness_mf, 'b', linewidth=2, label="Fitness Level (Normalized)")
plt.plot(x, deficiency_mf, 'r', linewidth=2, label="Deficiency")

plt.title("Membership Functions Used in Workout Recommender")
plt.xlabel("User Fitness Level (%)")
plt.ylabel("Membership Value")
plt.ylim(0, 1.1)
plt.legend()
plt.grid(True)
plt.show()
