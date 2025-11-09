# -------------------------------
# 🐺 Grey Wolf Optimizer with 2D Visualization
# -------------------------------

import numpy as np                # Used for numerical operations, arrays, and random number generation
import matplotlib.pyplot as plt    # Used to draw graphs and visualize wolves' movements on 2D plots
import time                        # Used to add small pauses between visualization frames


# -------------------------------
# Step 1: Define Objective Function
# -------------------------------
def objective_function(x):
    x1, x2 = x[0], x[1]                                # Extract values of x1 and x2 from input vector 'x'
    return x1**2 - x1*x2 + x2**2 + 2*x1 + 4*x2 + 3     # Return the calculated function value (fitness value)
                                                       # This function defines the problem we are minimizing


# -------------------------------
# Step 2: Grey Wolf Optimizer Function
# -------------------------------
def GWO(fitness_func, dim, lb, ub, N, Max_iter):
    # Initialize random positions of N wolves in 'dim' dimensions within given bounds [lb, ub]
    X = np.random.uniform(lb, ub, (N, dim))

    # Calculate fitness value of each wolf by applying the objective function
    fitness = np.apply_along_axis(fitness_func, 1, X)

    # Create empty arrays for top 3 wolves: Alpha (best), Beta (second best), Delta (third best)
    Alpha, Beta, Delta = np.zeros(dim), np.zeros(dim), np.zeros(dim)

    # Initialize their fitness as infinity (very large number) so any value will be better
    fAlpha, fBeta, fDelta = float("inf"), float("inf"), float("inf")

    # Identify the top 3 wolves based on initial fitness values
    for i in range(N):
        if fitness[i] < fAlpha:                      # If current wolf is better than Alpha
            fDelta, Delta = fBeta, Beta.copy()       # Move Beta's data to Delta
            fBeta, Beta = fAlpha, Alpha.copy()       # Move Alpha's data to Beta
            fAlpha, Alpha = fitness[i], X[i].copy()  # Set new Alpha as current wolf
        elif fitness[i] < fBeta:                     # If better than Beta but not Alpha
            fDelta, Delta = fBeta, Beta.copy()       # Move Beta to Delta
            fBeta, Beta = fitness[i], X[i].copy()    # Set current as new Beta
        elif fitness[i] < fDelta:                    # If better than Delta only
            fDelta, Delta = fitness[i], X[i].copy()  # Set current as new Delta

    # Create a grid of x1 and x2 values for visualization (for contour plotting)
    x1 = np.linspace(lb, ub, 200)                    # Create 200 evenly spaced values between lb and ub for x-axis
    x2 = np.linspace(lb, ub, 200)                    # Same for y-axis
    X1, X2 = np.meshgrid(x1, x2)                     # Create 2D grid of x1 and x2
    Z = objective_function([X1, X2])                 # Compute function values for each grid point to draw contours

    # -------------------------------
    # Main optimization loop
    # -------------------------------
    for t in range(Max_iter):
        a = 2 - 2 * (t / Max_iter)                   # Linearly reduce 'a' from 2 to 0 as iterations increase
                                                     # This controls exploration (wide search) and exploitation (fine search)

        # Loop through each wolf to update its position
        for i in range(N):
            for j in range(dim):                     # For each dimension (x1, x2)

                # --- Movement relative to Alpha (best wolf) ---
                r1, r2 = np.random.rand(), np.random.rand()    # Generate random numbers between 0 and 1
                A1 = 2 * a * r1 - a                            # Compute coefficient A1 (controls step direction/size)
                C1 = 2 * r2                                    # Compute coefficient C1 (controls attraction)
                D_alpha = abs(C1 * Alpha[j] - X[i, j])         # Distance between Alpha and current wolf
                X1_new = Alpha[j] - A1 * D_alpha               # New candidate position toward Alpha

                # --- Movement relative to Beta (second best) ---
                r1, r2 = np.random.rand(), np.random.rand()    # Generate new random numbers
                A2 = 2 * a * r1 - a                            # Compute A2 coefficient
                C2 = 2 * r2                                    # Compute C2 coefficient
                D_beta = abs(C2 * Beta[j] - X[i, j])           # Distance between Beta and current wolf
                X2_new = Beta[j] - A2 * D_beta                 # New candidate position toward Beta

                # --- Movement relative to Delta (third best) ---
                r1, r2 = np.random.rand(), np.random.rand()    # Generate new random numbers
                A3 = 2 * a * r1 - a                            # Compute A3 coefficient
                C3 = 2 * r2                                    # Compute C3 coefficient
                D_delta = abs(C3 * Delta[j] - X[i, j])         # Distance between Delta and current wolf
                X3_new = Delta[j] - A3 * D_delta               # New candidate position toward Delta

                # The new position is the average influence of Alpha, Beta, and Delta
                X[i, j] = (X1_new + X2_new + X3_new) / 3

            # Keep the wolf inside the search boundaries
            X[i] = np.clip(X[i], lb, ub)

        # Recalculate fitness of all wolves after updating their positions
        fitness = np.apply_along_axis(fitness_func, 1, X)

        # Update Alpha, Beta, and Delta wolves based on new fitness
        for i in range(N):
            if fitness[i] < fAlpha:                 # If current wolf is better than Alpha
                fDelta, Delta = fBeta, Beta.copy()  # Shift Beta to Delta
                fBeta, Beta = fAlpha, Alpha.copy()  # Shift Alpha to Beta
                fAlpha, Alpha = fitness[i], X[i].copy()  # Set new Alpha
            elif fitness[i] < fBeta:                # If better than Beta only
                fDelta, Delta = fBeta, Beta.copy()
                fBeta, Beta = fitness[i], X[i].copy()
            elif fitness[i] < fDelta:               # If better than Delta only
                fDelta, Delta = fitness[i], X[i].copy()

        # Display current iteration and best fitness value so far
        print(f"Iteration {t+1}/{Max_iter} --> Best Fitness: {fAlpha:.6f}")

        # -------------------------------
        # Step 3: Visualization (2D plot for each iteration)
        # -------------------------------
        plt.figure(figsize=(8, 6))                              # Create a new figure window
        plt.contourf(X1, X2, Z, levels=30, cmap='viridis')      # Draw a filled contour map of the objective function
        plt.colorbar(label="Objective Value")                    # Add color legend to show fitness values

        # Plot all wolves as white dots
        plt.scatter(X[:, 0], X[:, 1], color='white', edgecolor='black', label='Wolves')

        # Highlight Alpha, Beta, and Delta wolves in different colors and shapes
        plt.scatter(Alpha[0], Alpha[1], color='red', s=100, marker='*', label='Alpha (Best)')
        plt.scatter(Beta[0], Beta[1], color='orange', s=80, marker='^', label='Beta')
        plt.scatter(Delta[0], Delta[1], color='blue', s=80, marker='s', label='Delta')

        # Add chart title, axis labels, and legend
        plt.title(f"GWO Iteration {t+1}/{Max_iter}")
        plt.xlabel("x₁")                                       # Label x-axis
        plt.ylabel("x₂")                                       # Label y-axis
        plt.legend()                                           # Display the legend
        plt.tight_layout()                                     # Adjust layout for neat appearance

        # Show this iteration’s visualization
        plt.show()

        # Pause for a short time before showing next iteration
        time.sleep(0.8)

    # Return the best solution (Alpha's position and its fitness value)
    return Alpha, fAlpha


# -------------------------------
# Step 4: Main Program
# -------------------------------
print("🐺 Grey Wolf Optimizer (2D Iteration Visualization)")   # Display program title

# Take user input for number of wolves and number of iterations
N = int(input("Enter number of wolves (N): "))                 # Get number of wolves from user
Max_iter = int(input("Enter maximum iterations (Max_iter): ")) # Get number of iterations from user

# Define problem dimension and boundary limits
dim = 2              # Problem has two decision variables: x₁ and x₂
lb, ub = -5, 5       # Lower and upper bounds of the search area

# Run the Grey Wolf Optimizer function
best_pos, best_score = GWO(objective_function, dim, lb, ub, N, Max_iter)

# Display final optimization results
print("\nOptimization Completed!")                            # Notify user that optimization finished
print("Best Position (x₁, x₂):", best_pos)                    # Print best found coordinates
print("Best Fitness Value:", best_score)                      # Print minimum objective function value found
