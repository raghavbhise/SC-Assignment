# --- Dynamic Particle Swarm Optimization 2D Visualization (Minima at 1,1) ---
# Title comment indicating the purpose of the code:
# This program demonstrates Particle Swarm Optimization (PSO) with a 2D live visualization
# where the objective function's minimum is located at coordinates (1,1).

import numpy as np                      # Imports NumPy library for handling mathematical operations and arrays
import matplotlib.pyplot as plt          # Imports Matplotlib for plotting and visualization
from IPython.display import clear_output # Used to clear the previous output in the notebook for live updating
import time                              # Imports time library to add delays between frames for animation effect

# Objective function (Shifted Sphere) -> Minimum at (1,1)
def objective_function(x, y):            # Defines the objective function to minimize (accepts two variables x and y)
    return (x - 1)**2 + (y - 1)**2       # Calculates (x−1)² + (y−1)², a paraboloid with a global minimum at (1,1)
# The objective function tells how “good” or “bad” each position is (the smaller, the better).

# --- User Inputs ---
num_particles = int(input("Enter number of particles: "))  # Takes user input for total number of particles in the swarm
iterations = int(input("Enter number of iterations: "))    # Takes user input for number of iterations (loop count)

# PSO Parameters
w, c1, c2 = 0.5, 1.5, 1.5               # Sets PSO parameters: inertia weight (w), cognitive constant (c1) - how much it follows its own best position
                                        # Social constant (c2) - how much it follows the global best position

# Initialize particles randomly
x = np.random.uniform(-10, 10, num_particles)  # Randomly generates initial X positions for all particles between -10 and 10
y = np.random.uniform(-10, 10, num_particles)  # Randomly generates initial Y positions for all particles between -10 and 10
vx = np.zeros(num_particles)                   # Initializes velocity in X direction for all particles to 0
vy = np.zeros(num_particles)                   # Initializes velocity in Y direction for all particles to 0

# Personal and global bests
pbest_x, pbest_y = x.copy(), y.copy()          # Initially, each particle’s personal best position is its current position
pbest_val = objective_function(x, y)           # Computes the objective value for each particle’s personal best position
gbest_index = np.argmin(pbest_val)             # Finds the index of the particle with the smallest (best) objective value
gbest_x, gbest_y = pbest_x[gbest_index], pbest_y[gbest_index]  # Sets global best position to that particle’s coordinates

# --- PSO Loop with Live 2D Visualization ---
for t in range(iterations):                    # Starts the main PSO loop that runs for the specified number of iterations

    # Update velocity and position
    r1, r2 = np.random.rand(num_particles), np.random.rand(num_particles)  # Generates random factors r1 and r2 for stochastic behavior
    vx = w * vx + c1 * r1 * (pbest_x - x) + c2 * r2 * (gbest_x - x)       # Updates X velocity based on inertia, personal, and global attraction
    vy = w * vy + c1 * r1 * (pbest_y - y) + c2 * r2 * (gbest_y - y)       # Updates Y velocity similarly
    x += vx                                                                # Updates each particle’s X position using new velocity
    y += vy                                                                # Updates each particle’s Y position using new velocity

    # Evaluate new positions
    new_val = objective_function(x, y)          # Calculates the objective value for each particle’s new position
    better = new_val < pbest_val                # Checks which particles have found better (lower) objective values
    pbest_x[better] = x[better]                 # Updates X personal best for those improved particles
    pbest_y[better] = y[better]                 # Updates Y personal best for those improved particles
    pbest_val[better] = new_val[better]         # Updates personal best values for improved particles
    gbest_index = np.argmin(pbest_val)          # Finds the index of the new global best particle
    gbest_x, gbest_y = pbest_x[gbest_index], pbest_y[gbest_index]  # Updates global best position based on that particle

    # --- 2D Visualization ---
    clear_output(wait=True)                     # Clears previous frame for live animation
    plt.figure(figsize=(7,6))                   # Creates a new figure window with specified dimensions

    # Background contour map of objective function
    X = np.linspace(-10, 10, 200)               # Generates 200 evenly spaced values for X axis from -10 to 10
    Y = np.linspace(-10, 10, 200)               # Generates 200 evenly spaced values for Y axis from -10 to 10
    X, Y = np.meshgrid(X, Y)                    # Creates a coordinate grid (mesh) from X and Y values
    Z = objective_function(X, Y)                # Computes the objective value at each grid point (for contour plotting)
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')  # Draws filled contour map of function values using ‘viridis’ color map
    plt.colorbar(label='Objective Value')       # Adds a color bar showing the function value scale

    # Plot particles and global best
    plt.scatter(x, y, color='red', s=50, label='Particles')       # Plots all current particle positions as red dots
    plt.scatter(gbest_x, gbest_y, color='gold', marker='*', s=200, label='Global Best')  # Marks global best position with a gold star
    plt.scatter(1, 1, color='black', marker='x', s=100, label='True Minima (1,1)')       # Marks the actual known minimum at (1,1) with black X

    plt.title(f'PSO Iteration {t+1} / {iterations}')  # Displays current iteration number in the plot title
    plt.xlabel('X')                                  # Labels X-axis
    plt.ylabel('Y')                                  # Labels Y-axis
    plt.legend(loc='upper right')                    # Adds legend explaining markers and colors
    plt.xlim(-10, 10)                                # Sets X-axis limits for consistent view
    plt.ylim(-10, 10)                                # Sets Y-axis limits for consistent view
    plt.grid(True)                                   # Displays grid lines for easier visualization
    plt.show()                                       # Shows the current plot frame
    time.sleep(0.4)                                  # Waits 0.4 seconds before next iteration for animation effect

# After PSO loop completes
print(f"\nOptimization completed!")                  # Prints completion message
print(f"Global Best Position: ({gbest_x:.4f}, {gbest_y:.4f})")  # Prints best position found (formatted to 4 decimals)
print(f"Global Best Value: {objective_function(gbest_x, gbest_y):.4f}")  # Prints corresponding minimum objective function value
