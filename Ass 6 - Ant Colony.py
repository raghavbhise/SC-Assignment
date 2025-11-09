import numpy as np                                # Import NumPy for arrays and math operations (alias np)
import matplotlib.pyplot as plt                   # Import matplotlib plotting functions (alias plt)
import random                                     # Import Python's random module (not used directly here)
import pandas as pd                               # Import pandas for tabular display of the distance matrix (alias pd)

# -------------------- USER INPUTS --------------------
num_cities = int(input("Enter number of cities: "))     # Ask user number of cities; convert input to integer
num_ants = int(input("Enter number of ants: "))         # Ask user number of ants; convert input to integer
num_iterations = int(input("Enter number of iterations: "))  # Ask user how many iterations; convert to int

# Coordinates for each city (randomly generated)
cities = np.random.randint(0, 100, size=(num_cities, 2))  # Create num_cities random (x,y) coordinates between 0 and 99

# Show city coordinates
print("\n📍 City Coordinates (x, y):")                    # Print header for city list
for i, (x, y) in enumerate(cities):                      # Loop over cities with index i and coordinates (x,y)
    print(f"City {i}: ({x}, {y})")                        # Print each city's index and coordinates

# Compute distance matrix
dist_matrix = np.zeros((num_cities, num_cities))         # Initialize an NxN matrix of zeros to hold pairwise distances
for i in range(num_cities):                              # Loop over each city index i (rows)
    for j in range(num_cities):                          # Loop over each city index j (columns)
        if i != j:                                       # If not the same city (i != j)
            dist_matrix[i][j] = np.linalg.norm(cities[i] - cities[j])  # Compute Euclidean distance and store it
        else:                                            # If i == j (same city)
            dist_matrix[i][j] = np.inf                   # Set distance to infinity so ants won't travel to the same city

# Show cost/distance matrix with city labels
print("\n🧮 Distance (Cost) Matrix:")                      # Print header for distance matrix
df = pd.DataFrame(np.round(dist_matrix, 2),               # Create a DataFrame rounded to 2 decimals for readability
                  columns=[f"City {i}" for i in range(num_cities)],  # Column labels City 0, City 1, ...
                  index=[f"City {i}" for i in range(num_cities)])    # Row labels City 0, City 1, ...
print(df)                                                # Print the DataFrame (distance matrix)

# Choose starting city
start_city = int(input(f"\nEnter starting city (0 to {num_cities-1}): "))  # Ask user to choose starting city index
if start_city < 0 or start_city >= num_cities:          # Validate the starting city input
    raise ValueError("Invalid starting city! Must be between 0 and number of cities - 1.")  # Raise error if invalid

# -------------------- ACO PARAMETERS --------------------
alpha = 1      # how much ants trust pheromone
beta = 5       # how much ants care about distance
evaporation = 0.5   # Fraction of pheromone that evaporates each iteration (0 < evaporation < 1)
Q = 100        # Pheromone deposit scaling factor (higher Q => larger pheromone deposits)

# Initialize pheromone matrix
pheromone = np.ones((num_cities, num_cities))           # Start with pheromone value 1 on every edge (NxN matrix)

# -------------------- ACO MAIN LOOP --------------------
best_distance = float('inf')                            # Initialize best distance as infinity (so first solution beats it)
best_path = None                                        # Initialize best path variable

for iteration in range(num_iterations):                 # Repeat algorithm for given number of iterations
    all_paths = []                                      # List to store each ant's path this iteration
    all_distances = []                                  # List to store each ant's total distance this iteration

    for ant in range(num_ants):                         # For each ant simulate a tour
        visited = [start_city]                          # Start the ant at the chosen starting city
        current_city = start_city                       # Set current city to starting city

        # Build the path
        for k in range(num_cities - 1):                 # Repeat until the ant has visited all other cities
            unvisited = [c for c in range(num_cities) if c not in visited]  # List of cities not yet visited

            # Calculate probabilities based on pheromone and distance
            probabilities = []                          # Prepare a list to hold attractiveness scores
            for next_city in unvisited:                 # For every possible next city
                pher = pheromone[current_city][next_city] ** alpha   # Pheromone influence (raised to alpha)
                dist = (1 / dist_matrix[current_city][next_city]) ** beta  # Distance influence as 1/d^beta
                probabilities.append(pher * dist)      # Combined attractiveness score for that next_city

            probabilities = np.array(probabilities)    # Convert list to NumPy array for math operations
            probabilities /= probabilities.sum()       # Normalize to sum to 1 (becomes probabilities)

            # Choose next city based on probability
            next_city = np.random.choice(unvisited, p=probabilities)  # Randomly pick next city with those probabilities
            visited.append(next_city)                  # Mark chosen city as visited (append to route)
            current_city = next_city                   # Move the ant to that city

        visited.append(start_city)                      # After visiting all cities, return to the starting city to complete the tour
        total_distance = sum(dist_matrix[visited[i]][visited[i + 1]] for i in range(num_cities))  # Sum distances along the tour
        all_paths.append(visited)                       # Save this ant's path
        all_distances.append(total_distance)            # Save this ant's path total distance

    # Update pheromones (evaporation + deposit)
    pheromone *= (1 - evaporation)                      # Evaporate pheromone from all edges (reduce pheromone values)
    for i, path in enumerate(all_paths):                # For each ant's path, deposit pheromone proportional to quality
        for j in range(num_cities):                     # For each edge in the ant's path
            a, b = path[j], path[j + 1]                 # Edge goes from city a to city b
            pheromone[a][b] += Q / all_distances[i]    # Add pheromone inversely proportional to path length (shorter path => more pheromone)

    # Track best path
    min_distance = min(all_distances)                   # Best distance among ants this iteration
    min_path = all_paths[np.argmin(all_distances)]      # Path corresponding to that best distance

    if min_distance < best_distance:                    # If this iteration produced a new global best
        best_distance = min_distance                    # Update global best distance
        best_path = min_path                             # Update global best path

    print(f"Iteration {iteration + 1}: 🐜 Best Distance = {best_distance:.2f} | Path = {best_path}")  # Print progress for this iteration

# -------------------- FINAL VISUALIZATION --------------------
print("\n🏁 Final Best Path Found:", " → ".join(map(str, best_path)))  # Print readable best path sequence
print(f"📏 Total Distance (Cost): {best_distance:.2f}")               # Print the total distance of the best path

plt.figure(figsize=(8, 6))                                # Create a figure of size 8x6 inches for plotting
plt.scatter(cities[:, 0], cities[:, 1], color='red', s=100)  # Plot city coordinates as red dots
for i, (x, y) in enumerate(cities):                       # For each city, label its index near the point
    plt.text(x + 1, y + 1, str(i), fontsize=12)           # Draw the city index slightly offset from the point

# Draw best path
for i in range(len(best_path) - 1):                       # For each segment in the best path
    start = best_path[i]                                  # Starting city index of this segment
    end = best_path[i + 1]                                # Ending city index of this segment
    plt.plot([cities[start, 0], cities[end, 0]],          # Plot a line from start city x to end city x
             [cities[start, 1], cities[end, 1]],          # and from start city y to end city y
             'b-', linewidth=2)                           # Use a blue solid line with width 2

plt.title(f"Best Path Visualization (Distance = {best_distance:.2f})")  # Title showing the best distance
plt.xlabel("X Coordinate")                                 # Label for x-axis
plt.ylabel("Y Coordinate")                                 # Label for y-axis
plt.grid(True)                                             # Show grid for easier reading
plt.show()                                                 # Display the plot window
