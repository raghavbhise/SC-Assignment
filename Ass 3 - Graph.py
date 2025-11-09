import numpy as np
import matplotlib.pyplot as plt


# Triangular Membership Function
def triangularMF(x, a, b, c):
    return np.where(
        (x <= a) | (x >= c), 0.0,
        np.where(x == b, 1.0,
                 np.where((x > a) & (x < b), (x - a) / (b - a), (c - x) / (c - b)))
    )
# X range (normalized 0–1)
x = np.linspace(0, 1, 200)
# Membership functions for Moisture, Days, Temp (normalized)
moisture_mf = triangularMF(x, 0.2, 0.5, 0.8)
days_mf     = triangularMF(x, 0.1, 0.5, 1.0)
temp_mf     = triangularMF(x, 0.0, 0.5, 1.0)
# Plot
plt.figure(figsize=(10,6))
plt.plot(x, moisture_mf, label="Moisture (0.2, 0.5, 0.8)", linewidth=2)
plt.plot(x, days_mf,     label="Days (0.1, 0.5, 1.0)", linewidth=2)
plt.plot(x, temp_mf,     label="Temperature (0.0, 0.5, 1.0)", linewidth=2)
plt.title("Triangular Membership Functions (Normalized)", fontsize=14)
plt.xlabel("Normalized Input (0–1)")
plt.ylabel("Membership Degree (0–1)")
plt.legend()
plt.grid(True)
plt.show()


# Trapezoidal Membership Function
def trapezoidalMF(x, a, b, c, d):
    return np.where(
        (x <= a) | (x >= d), 0.0,
        np.where((x >= b) & (x <= c), 1.0,
                 np.where((x > a) & (x < b), (x - a) / (b - a), (d - x) / (d - c)))
    )
# X range (normalized 0–1)
x = np.linspace(0, 1, 200)
# Membership functions for Moisture, Days, Temp (normalized)
moisture_mf = trapezoidalMF(x, 0.2, 0.4, 0.6, 0.8)
days_mf     = trapezoidalMF(x, 0.0, 0.3, 0.7, 1.0)
temp_mf     = trapezoidalMF(x, 0.0, 0.3, 0.7, 1.0)
# Plot
plt.figure(figsize=(10,6))
plt.plot(x, moisture_mf, label="Moisture (0.2, 0.4, 0.6, 0.8)", linewidth=2)
plt.plot(x, days_mf,     label="Days (0.0, 0.3, 0.7, 1.0)", linewidth=2)
plt.plot(x, temp_mf,     label="Temperature (0.0, 0.3, 0.7, 1.0)", linewidth=2)
plt.title("Trapezoidal Membership Functions (Normalized)", fontsize=14)
plt.xlabel("Normalized Input (0–1)")
plt.ylabel("Membership Degree (0–1)")
plt.legend()
plt.grid(True)
plt.show()


# Gaussian Membership Function
def gaussianMF(x, c, sigma):
    return np.exp(-0.5 * ((x - c) / sigma) ** 2)
# X range (normalized 0–1)
x = np.linspace(0, 1, 200)
# Membership functions for Moisture, Days, Temp (normalized)
moisture_mf = gaussianMF(x, 0.5, 0.15)
days_mf     = gaussianMF(x, 0.5, 0.2)
temp_mf     = gaussianMF(x, 0.5, 0.2)
# Plot
plt.figure(figsize=(10,6))
plt.plot(x, moisture_mf, label="Moisture (c=0.5, σ=0.15)", linewidth=2)
plt.plot(x, days_mf,     label="Days (c=0.5, σ=0.2)", linewidth=2)
plt.plot(x, temp_mf,     label="Temperature (c=0.5, σ=0.2)", linewidth=2)
plt.title("Gaussian Membership Functions (Normalized)", fontsize=14)
plt.xlabel("Normalized Input (0–1)")
plt.ylabel("Membership Degree (0–1)")
plt.legend()
plt.grid(True)
plt.show()


# Generalized Bell Membership Function
def bellMF(x, a, b, c):
    return 1 / (1 + np.abs((x - c) / a) ** (2 * b))
# X range (normalized 0–1)
x = np.linspace(0, 1, 200)
# Membership functions for Moisture, Days, Temp (normalized)
moisture_mf = bellMF(x, 0.1, 2, 0.5)
days_mf     = bellMF(x, 0.2, 2, 0.5)
temp_mf     = bellMF(x, 0.2, 2, 0.5)
# Plot
plt.figure(figsize=(10,6))
plt.plot(x, moisture_mf, label="Moisture (a=0.1, b=2, c=0.5)", linewidth=2)
plt.plot(x, days_mf,     label="Days (a=0.2, b=2, c=0.5)", linewidth=2)
plt.plot(x, temp_mf,     label="Temperature (a=0.2, b=2, c=0.5)", linewidth=2)
plt.title("Generalized Bell-Shaped Membership Functions", fontsize=14)
plt.xlabel("Normalized Input (0–1)")
plt.ylabel("Membership Degree (0–1)")
plt.legend()
plt.grid(True)
plt.show()


# Sigmoid Membership Function
def sigmoidMF(x, a, c):
    return 1 / (1 + np.exp(-a * (x - c)))
# X range (normalized 0–1)
x = np.linspace(0, 1, 200)
# Membership functions for Moisture, Days, Temp (normalized)
moisture_mf = sigmoidMF(x, 10, 0.5)
days_mf     = sigmoidMF(x, 10, 0.5)
temp_mf     = sigmoidMF(x, 10, 0.5)
# Plot
plt.figure(figsize=(10,6))
plt.plot(x, moisture_mf, label="Moisture (a=10, c=0.5)", linewidth=2)
plt.plot(x, days_mf,     label="Days (a=10, c=0.5)", linewidth=2)
plt.plot(x, temp_mf,     label="Temperature (a=10, c=0.5)", linewidth=2)
plt.title("Sigmoidal Membership Functions", fontsize=14)
plt.xlabel("Normalized Input (0–1)")
plt.ylabel("Membership Degree (0–1)")
plt.legend()
plt.grid(True)
plt.show()


# Left-Right MF definition (Gaussian left, Sigmoid right type)
def leftRightMF(x, a, b, c):
    # Left side (Gaussian)
    left = np.exp(-((x - b) ** 2) / (2 * (a ** 2)))
    # Right side (Sigmoid)
    right = 1 / (1 + np.exp(-10 * (x - c)))
    return np.minimum(left, right)
# X range (normalized 0–1)
x = np.linspace(0, 1, 200)
# Membership functions
moisture_mf = leftRightMF(x, 0.2, 0.4, 0.3)
days_mf     = leftRightMF(x, 0.2, 0.5, 0.2)
temp_mf     = leftRightMF(x, 0.2, 0.5, 0.2)
# Plt
plt.figure(figsize=(10,6))
plt.plot(x, moisture_mf, label="Moisture (a=0.2, b=0.4, c=0.3)", linewidth=2)
plt.plot(x, days_mf,     label="Days (a=0.2, b=0.5, c=0.2)", linewidth=2)
plt.plot(x, temp_mf,     label="Temperature (a=0.2, b=0.5, c=0.2)", linewidth=2)
plt.title("Left-Right Membership Functions", fontsize=14)
plt.xlabel("Normalized Input (0–1)")
plt.ylabel("Membership Degree (0–1)")
plt.legend()
plt.grid(True)
plt.show()


# Pi-shaped MF definition
def piMF(x, a, b, c, d):
    y = np.zeros_like(x)
    # Rising part (S-shaped between a and b)
    idx1 = (x >= a) & (x <= b)
    y[idx1] = 2 * ((x[idx1] - a) / (b - a))**2
    idx2 = (x > b) & (x <= (a+b)/2 + (b-a)/2)  # midpoint region
    y[idx2] = 1 - 2 * ((x[idx2] - b) / (b - a))**2
    # Middle plateau (between b and c)
    y[(x > b) & (x < c)] = 1
    # Falling part (Z-shaped between c and d)
    idx3 = (x >= c) & (x <= d)
    y[idx3] = 1 - 2 * ((x[idx3] - c) / (d - c))**2
    idx4 = (x > (c+d)/2) & (x < d)
    y[idx4] = 2 * ((x[idx4] - d) / (d - c))**2
    return np.clip(y, 0, 1)
# X range (normalized 0–1)
x = np.linspace(0, 1, 500)
# Membership functions
moisture_mf = piMF(x, 0.2, 0.4, 0.6, 0.8)
days_mf     = piMF(x, 0.0, 0.3, 0.7, 1.0)
temp_mf     = piMF(x, 0.0, 0.3, 0.7, 1.0)
# Plot
plt.figure(figsize=(10,6))
plt.plot(x, moisture_mf, label="Moisture (0.2, 0.4, 0.6, 0.8)", linewidth=2)
plt.plot(x, days_mf,     label="Days (0.0, 0.3, 0.7, 1.0)", linewidth=2)
plt.plot(x, temp_mf,     label="Temperature (0.0, 0.3, 0.7, 1.0)", linewidth=2)
plt.title("Π-shaped Membership Functions", fontsize=14)
plt.xlabel("Normalized Input (0–1)")
plt.ylabel("Membership Degree (0–1)")
plt.legend()
plt.grid(True)
plt.show()


# Open Left MF definition
def openLeftMF(x, a, b):
    y = np.ones_like(x)
    # Slope between a and b
    idx = (x >= a) & (x <= b)
    y[idx] = (b - x[idx]) / (b - a)
    # Right side (after b → 0)
    y[x > b] = 0
    return np.clip(y, 0, 1)
# X range (normalized 0–1)
x = np.linspace(0, 1, 500)
# Membership functions
moisture_mf = openLeftMF(x, 0.2, 0.6)
days_mf     = openLeftMF(x, 0.3, 0.7)
temp_mf     = openLeftMF(x, 0.3, 0.7)
# Plot
plt.figure(figsize=(10,6))
plt.plot(x, moisture_mf, label="Moisture (0.2, 0.6)", linewidth=2)
plt.plot(x, days_mf,     label="Days (0.3, 0.7)", linewidth=2)
plt.plot(x, temp_mf,     label="Temperature (0.3, 0.7)", linewidth=2)
plt.title("Open Left Membership Functions", fontsize=14)
plt.xlabel("Normalized Input (0–1)")
plt.ylabel("Membership Degree (0–1)")
plt.legend()
plt.grid(True)
plt.show()

# Open Right MF definition
def openRightMF(x, a, b):
    y = np.zeros_like(x)
    # Slope between a and b
    idx = (x >= a) & (x <= b)
    y[idx] = (x[idx] - a) / (b - a)
    # Right side (after b → 1)
    y[x > b] = 1
    return np.clip(y, 0, 1)
# X range (normalized 0–1)
x = np.linspace(0, 1, 500)
# Membership functions
moisture_mf = openRightMF(x, 0.2, 0.6)
days_mf     = openRightMF(x, 0.3, 0.7)
temp_mf     = openRightMF(x, 0.3, 0.7)
# Plot
plt.figure(figsize=(10,6))
plt.plot(x, moisture_mf, label="Moisture (0.2, 0.6)", linewidth=2)
plt.plot(x, days_mf,     label="Days (0.3, 0.7)", linewidth=2)
plt.plot(x, temp_mf,     label="Temperature (0.3, 0.7)", linewidth=2)
plt.title("Open Right Membership Functions", fontsize=14)
plt.xlabel("Normalized Input (0–1)")
plt.ylabel("Membership Degree (0–1)")
plt.legend()
plt.grid(True)
plt.show()


# S-Shaped MF definition
def sShapedMF(x, a, b):
    y = np.zeros_like(x)
    idx1 = (x >= a) & (x <= (a+b)/2)
    y[idx1] = 2 * ((x[idx1] - a) / (b - a))**2
    idx2 = (x > (a+b)/2) & (x <= b)
    y[idx2] = 1 - 2 * ((b - x[idx2]) / (b - a))**2
    y[x > b] = 1
    return np.clip(y, 0, 1)
# X range (normalized 0–1)
x = np.linspace(0, 1, 500)
# Membership functions
moisture_mf = sShapedMF(x, 0.2, 0.8)
days_mf     = sShapedMF(x, 0.0, 1.0)
temp_mf     = sShapedMF(x, 0.0, 1.0)
# Plot
plt.figure(figsize=(10,6))
plt.plot(x, moisture_mf, label="Moisture (0.2, 0.8)", linewidth=2)
plt.plot(x, days_mf,     label="Days (0.0, 1.0)", linewidth=2)
plt.plot(x, temp_mf,     label="Temperature (0.0, 1.0)", linewidth=2)
plt.title("S-Shaped Membership Functions", fontsize=14)
plt.xlabel("Normalized Input (0–1)")
plt.ylabel("Membership Degree (0–1)")
plt.legend()
plt.grid(True)
plt.show()