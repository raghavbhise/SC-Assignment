# Import necessary libraries
import numpy as np                    # For numerical operations, arrays, and mathematical functions
import random                         # For generating random numbers (used in crossover/mutation decisions)
import matplotlib.pyplot as plt       # For plotting the function and population evolution

# ------------------------------------------------------------
# Objective Function (Rastrigin) — we are minimizing this
# ------------------------------------------------------------
def objective_function(x):
    # Rastrigin function: a common benchmark for optimization problems
    # It has many local minima, making it challenging for optimization algorithms
    return x**2 - 10 * np.cos(2 * np.pi * x) + 10

# Convert objective value to fitness (since GA maximizes fitness)
def fitness_function(x):
    # Compute objective value first
    f = objective_function(x)
    # Convert it to fitness by taking reciprocal (smaller f → larger fitness)
    # Added a small constant (1e-6) to prevent division by zero
    return 1 / (f + 1e-6)


# ------------------------------------------------------------
# GA PARAMETERS (USER INPUT)
# ------------------------------------------------------------
POP_SIZE = int(input("Enter Population Size: "))         # Number of individuals in the population
GENS = int(input("Enter Number of Generations: "))       # Number of iterations for evolution
CROSS_RATE = float(input("Enter Crossover Rate (0–1): "))# Probability that crossover occurs
MUT_RATE = float(input("Enter Mutation Rate (0–1): "))   # Probability that mutation occurs
X_BOUND = [-5.12, 5.12]                                 # Search space limits for Rastrigin function
TEMPERATURE = 1.0                                        # Temperature parameter for Boltzmann selection


# ------------------------------------------------------------
# INITIALIZE POPULATION
# ------------------------------------------------------------
def initialize_population():
    # Create an initial population of random values within defined bounds
    return np.random.uniform(X_BOUND[0], X_BOUND[1], POP_SIZE)


# ------------------------------------------------------------
# SELECTION METHODS
# ------------------------------------------------------------

# Canonical Selection: selects individuals based on proportional fitness probability
def canonical_selection(pop, fitness):
    probs = fitness / np.sum(fitness)                    # Normalize fitness to probabilities
    return np.random.choice(pop, size=POP_SIZE, p=probs) # Randomly select individuals using probabilities

# Roulette Wheel Selection: similar to canonical, uses fitness proportionate selection
def roulette_selection(pop, fitness):
    probs = fitness / np.sum(fitness)
    return np.random.choice(pop, size=POP_SIZE, p=probs)

# Rank-Based Selection: assigns probabilities based on rank rather than actual fitness values
def rank_based_selection(pop, fitness):
    ranks = np.argsort(np.argsort(fitness))              # Rank individuals
    probs = (ranks + 1) / np.sum(ranks + 1)              # Convert ranks to probabilities
    return np.random.choice(pop, size=POP_SIZE, p=probs)

# Tournament Selection: picks best among randomly chosen individuals (tournament of k)
def tournament_selection(pop, fitness, k=3):
    selected = []
    for _ in range(POP_SIZE):
        idx = np.random.randint(0, POP_SIZE, k)          # Choose k random individuals
        best = idx[np.argmax(fitness[idx])]              # Pick the one with highest fitness
        selected.append(pop[best])
    return np.array(selected)

# Steady-State Selection: keeps top elite individuals, rest filled with random others
def steady_state_selection(pop, fitness):
    sorted_indices = np.argsort(-fitness)                # Sort fitness in descending order
    elite_count = int(POP_SIZE * 0.2)                    # Keep top 20% as elites
    elites = pop[sorted_indices[:elite_count]]           # Select elites
    others = np.random.choice(pop, POP_SIZE - elite_count)
    return np.concatenate((elites, others))              # Combine elites and random others

# Boltzmann Selection: uses exponential probabilities controlled by temperature
def boltzmann_selection(pop, fitness, temperature):
    probs = np.exp(fitness / temperature) / np.sum(np.exp(fitness / temperature))
    return np.random.choice(pop, size=POP_SIZE, p=probs)


# ------------------------------------------------------------
# CROSSOVER METHODS
# ------------------------------------------------------------

# Single-point (blend) crossover: mix two parents linearly using random weight alpha
def single_point_crossover(p1, p2):
    if random.random() < CROSS_RATE:                     # Perform crossover based on probability
        alpha = random.random()                          # Random mixing factor
        return alpha * p1 + (1 - alpha) * p2, (1 - alpha) * p1 + alpha * p2
    return p1, p2                                        # Return unchanged parents if no crossover

# Two-point crossover: similar but uses two random mix coefficients
def two_point_crossover(p1, p2):
    if random.random() < CROSS_RATE:
        beta1, beta2 = np.random.uniform(0, 1, 2)
        return beta1 * p1 + (1 - beta1) * p2, beta2 * p2 + (1 - beta2) * p1
    return p1, p2

# Arithmetic crossover: generates children as weighted averages of parents
def arithmetic_crossover(p1, p2):
    if random.random() < CROSS_RATE:
        alpha = random.uniform(0, 1)
        return alpha * p1 + (1 - alpha) * p2, (1 - alpha) * p1 + alpha * p2
    return p1, p2

# Blend Crossover (BLX-α): expands search space by extending beyond parent range
def blend_crossover_blx(p1, p2, alpha=0.5):
    if random.random() < CROSS_RATE:
        d = abs(p1 - p2)
        low = min(p1, p2) - alpha * d
        high = max(p1, p2) + alpha * d
        c1 = np.random.uniform(low, high)
        c2 = np.random.uniform(low, high)
        return c1, c2
    return p1, p2

# Heuristic crossover: biases child toward better parent based on fitness
def heuristic_crossover(p1, p2, f1, f2):
    if random.random() < CROSS_RATE:
        if f1 > f2:
            child = p1 + random.random() * (p1 - p2)
        else:
            child = p2 + random.random() * (p2 - p1)
        return np.clip(child, X_BOUND[0], X_BOUND[1]), child
    return p1, p2


# ------------------------------------------------------------
# MUTATION METHODS
# ------------------------------------------------------------

# Uniform mutation: replace with a random value within bounds
def uniform_mutation(x):
    if random.random() < MUT_RATE:
        x = np.random.uniform(X_BOUND[0], X_BOUND[1])
    return np.clip(x, X_BOUND[0], X_BOUND[1])

# Gaussian mutation: add small random noise from normal distribution
def gaussian_mutation(x):
    if random.random() < MUT_RATE:
        x += np.random.normal(0, 0.1)
    return np.clip(x, X_BOUND[0], X_BOUND[1])

# Boundary mutation: randomly jump to lower or upper bound
def boundary_mutation(x):
    if random.random() < MUT_RATE:
        x = random.choice(X_BOUND)
    return x

# Non-uniform mutation: mutation strength decreases with generation number
def non_uniform_mutation(x, gen, max_gen):
    if random.random() < MUT_RATE:
        delta = (1 - gen / max_gen)**2                  # Mutation range shrinks over time
        x += np.random.uniform(-delta, delta)
    return np.clip(x, X_BOUND[0], X_BOUND[1])

# Creep mutation: small gradual change added to gene
def creep_mutation(x):
    if random.random() < MUT_RATE:
        x += random.uniform(-0.05, 0.05)
    return np.clip(x, X_BOUND[0], X_BOUND[1])


# ------------------------------------------------------------
# METHOD SELECTION MENU
# ------------------------------------------------------------
selection_methods = {                                   # Dictionary mapping for selection choices
    "1": canonical_selection,
    "2": roulette_selection,
    "3": rank_based_selection,
    "4": tournament_selection,
    "5": steady_state_selection,
    "6": boltzmann_selection
}

crossover_methods = {                                   # Dictionary for crossover method mapping
    "1": single_point_crossover,
    "2": two_point_crossover,
    "3": arithmetic_crossover,
    "4": blend_crossover_blx,
    "5": heuristic_crossover
}

mutation_methods = {                                   # Dictionary for mutation method mapping
    "1": uniform_mutation,
    "2": gaussian_mutation,
    "3": boundary_mutation,
    "4": non_uniform_mutation,
    "5": creep_mutation
}

# Display menu and take user choices
print("\nSelection Methods:\n1. Canonical\n2. Roulette\n3. Rank-Based\n4. Tournament\n5. Steady-State\n6. Boltzmann")
sel_choice = input("Select Selection Method (1–6): ")
sel_method = selection_methods[sel_choice]

print("\nCrossover Methods:\n1. Single-Point\n2. Two-Point\n3. Arithmetic\n4. Blend BLX\n5. Heuristic")
cross_choice = input("Select Crossover Method (1–5): ")
cross_method = crossover_methods[cross_choice]

print("\nMutation Methods:\n1. Uniform\n2. Gaussian\n3. Boundary\n4. Non-Uniform\n5. Creep")
mut_choice = input("Select Mutation Method (1–5): ")
mut_method = mutation_methods[mut_choice]


# ------------------------------------------------------------
# MAIN GA EXECUTION
# ------------------------------------------------------------
population = initialize_population()                   # Generate initial random population

for gen in range(GENS):                                # Loop for all generations
    fitness = fitness_function(population)             # Evaluate fitness of each individual

    # --- SELECTION ---
    if sel_method == boltzmann_selection:
        selected = sel_method(population, fitness, TEMPERATURE)  # Boltzmann needs temperature
    else:
        selected = sel_method(population, fitness)               # Other methods just need pop & fitness

    # --- CROSSOVER ---
    children = []
    for i in range(0, POP_SIZE, 2):                    # Pair individuals for crossover
        p1, p2 = selected[i], selected[(i + 1) % POP_SIZE]
        f1, f2 = fitness[i], fitness[(i + 1) % POP_SIZE]
        # If heuristic crossover, pass fitness values too
        c1, c2 = cross_method(p1, p2) if cross_method != heuristic_crossover else cross_method(p1, p2, f1, f2)
        children.extend([c1, c2])                      # Add children to new population

    # --- MUTATION ---
    for i in range(len(children)):
        # Apply mutation method; for non-uniform mutation, include generation info
        children[i] = mut_method(children[i]) if mut_method != non_uniform_mutation else mut_method(children[i], gen, GENS)

    population = np.array(children)                    # Replace old population with new generation

    # --- TRACK BEST RESULT ---
    f_values = objective_function(population)          # Evaluate objective values
    best_idx = np.argmin(f_values)                     # Get index of best (minimum) value
    print(f"Gen {gen+1}: Best f(X) = {f_values[best_idx]:.5f}")  # Display progress


# ------------------------------------------------------------
# FINAL RESULT
# ------------------------------------------------------------
best_x = population[np.argmin(objective_function(population))]  # Find best solution from final population
best_y = objective_function(best_x)
print("\nOptimal Solution Found:")
print(f"Best X = {best_x:.4f}, Minimum f(X) = {best_y:.4f}")   # Display final optimized result

# Plot final population and function curve
x = np.linspace(X_BOUND[0], X_BOUND[1], 400)          # Generate x-axis values for smooth curve
plt.plot(x, objective_function(x), label="Rastrigin Function")  # Plot actual function curve
plt.scatter(population, objective_function(population), color='red', label="Final Population")  # Plot individuals
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title(f"GA Optimization\nSelection={sel_method.__name__}, Crossover={cross_method.__name__}, Mutation={mut_method.__name__}")
plt.legend()
plt.show()                                            # Display the final graph
