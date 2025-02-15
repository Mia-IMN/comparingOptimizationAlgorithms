import numpy as np
import time
import matplotlib.pyplot as plt
from prettytable import PrettyTable


# -------------------------------
# Define the Objective Function
# -------------------------------
def sphere(x):
    return np.sum(x ** 2)


# -------------------------------
# 1. Particle Swarm Optimization (PSO)
# -------------------------------
def optimize_pso(dim, max_iter=100):
    swarm_size = 30
    positions = np.random.uniform(-100, 100, (swarm_size, dim))
    velocities = np.random.uniform(-1, 1, (swarm_size, dim))
    personal_best = positions.copy()
    personal_best_scores = np.apply_along_axis(sphere, 1, positions)
    gbest_index = np.argmin(personal_best_scores)
    global_best = positions[gbest_index].copy()

    start_time = time.perf_counter()
    for _ in range(max_iter):
        r1 = np.random.rand(swarm_size, dim)
        r2 = np.random.rand(swarm_size, dim)
        velocities = velocities + 2.0 * r1 * (personal_best - positions) + 2.0 * r2 * (global_best - positions)
        positions = positions + velocities

        scores = np.apply_along_axis(sphere, 1, positions)
        improved = scores < personal_best_scores
        personal_best[improved] = positions[improved]
        personal_best_scores[improved] = scores[improved]
        current_gbest_index = np.argmin(personal_best_scores)
        if personal_best_scores[current_gbest_index] < sphere(global_best):
            global_best = personal_best[current_gbest_index].copy()
    end_time = time.perf_counter()
    latency = end_time - start_time
    return global_best, latency


# -------------------------------
# 2. Genetic Algorithm (GA)
# -------------------------------
def optimize_ga(dim, max_iter=100):
    pop_size = 30
    mutation_rate = 0.1
    population = np.random.uniform(-100, 100, (pop_size, dim))

    start_time = time.perf_counter()
    for _ in range(max_iter):
        fitness = np.apply_along_axis(sphere, 1, population)
        selected = []
        for _ in range(pop_size):
            i, j = np.random.choice(pop_size, 2, replace=False)
            winner = population[i] if fitness[i] < fitness[j] else population[j]
            selected.append(winner)
        selected = np.array(selected)
        offspring = []
        for i in range(0, pop_size, 2):
            parent1 = selected[i]
            parent2 = selected[(i + 1) % pop_size]
            cp = np.random.randint(1, dim)
            child1 = np.concatenate([parent1[:cp], parent2[cp:]])
            child2 = np.concatenate([parent2[:cp], parent1[cp:]])
            offspring.extend([child1, child2])
        offspring = np.array(offspring)[:pop_size]
        mutation_mask = np.random.rand(pop_size, dim) < mutation_rate
        offspring = offspring + mutation_mask * np.random.uniform(-1, 1, (pop_size, dim))
        population = offspring
    fitness = np.apply_along_axis(sphere, 1, population)
    best_solution = population[np.argmin(fitness)]
    end_time = time.perf_counter()
    latency = end_time - start_time
    return best_solution, latency


# -------------------------------
# 3. Firefly Algorithm
# -------------------------------
# def optimize_firefly(dim, max_iter=100):
#     n_fireflies = 30
#     alpha = 0.2  # Randomness parameter
#     beta0 = 1.0  # Attractiveness at distance 0
#     gamma = 1.0 / dim  # Light absorption coefficient
#
#     fireflies = np.random.uniform(-100, 100, (n_fireflies, dim))
#
#     start_time = time.perf_counter()
#     for _ in range(max_iter):
#         brightness = 1.0 / (1.0 + np.apply_along_axis(sphere, 1, fireflies))
#         for i in range(n_fireflies):
#             for j in range(n_fireflies):
#                 if brightness[j] > brightness[i]:
#                     r = np.linalg.norm(fireflies[i] - fireflies[j])
#                     beta = beta0 * np.exp(-gamma * r ** 2)
#                     fireflies[i] = fireflies[i] + beta * (fireflies[j] - fireflies[i]) + alpha * (
#                                 np.random.rand(dim) - 0.5)
#     end_time = time.perf_counter()
#     latency = end_time - start_time
#     fitness = np.apply_along_axis(sphere, 1, fireflies)
#     best_solution = fireflies[np.argmin(fitness)]
#     return best_solution, latency


# -------------------------------
# 4. Continuous Ant Colony Optimization (ACO)
# -------------------------------
def optimize_aco(dim, max_iter=100):
    n_ants = 30
    solutions = np.random.uniform(-100, 100, (n_ants, dim))
    start_time = time.perf_counter()
    for _ in range(max_iter):
        fitness = np.apply_along_axis(sphere, 1, solutions)
        pheromone = 1.0 / (1.0 + fitness)
        best_idx = np.argmin(fitness)
        best_solution = solutions[best_idx]
        for i in range(n_ants):
            noise_scale = 1.0 / (pheromone[i] + 1e-6)
            solutions[i] = best_solution + np.random.randn(dim) * noise_scale
    end_time = time.perf_counter()
    latency = end_time - start_time
    fitness = np.apply_along_axis(sphere, 1, solutions)
    best_solution = solutions[np.argmin(fitness)]
    return best_solution, latency


# -------------------------------
# Running the Algorithms and Collecting Latency Data
# -------------------------------
dims = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
results = {'PSO': [], 'ACO': [], 'Firefly': [], 'Genetic': []}

print("Collecting latency data...")
for d in dims:
    print(f"Dimension (tasks): {d}")
    _, t_pso = optimize_pso(d)
    _, t_aco = optimize_aco(d)
    # _, t_firefly = optimize_firefly(d)
    _, t_ga = optimize_ga(d)
    results['PSO'].append(t_pso)
    results['ACO'].append(t_aco)
    # results['Firefly'].append(t_firefly)
    results['Genetic'].append(t_ga)
    print(f"  PSO: {t_pso:.4f}s, ACO: {t_aco:.4f}s, GA: {t_ga:.4f}s") # Firefly: {t_firefly:.4f}s

# -------------------------------
# Displaying the Results with PrettyTable
# -------------------------------
table = PrettyTable()
table.field_names = ["Dimension (Tasks)", "PSO (s)", "ACO (s)", "Genetic (s)"]

for i, d in enumerate(dims):
    table.add_row([
        d,
        f"{results['PSO'][i]:.4f}",
        f"{results['ACO'][i]:.4f}",
        # f"{results['Firefly'][i]:.4f}",
        f"{results['Genetic'][i]:.4f}"
    ])

print("\nLatency Comparison Table:")
print(table)

# -------------------------------
# Creating a Grouped Bar Chart Using the Actual Measured Latencies
# -------------------------------
x = np.arange(len(dims))
width = 0.2  # Width of each bar

plt.figure(figsize=(12, 6))
plt.bar(x - 1.5 * width, results['PSO'], width, label='PSO', color = '0.75')
plt.bar(x - 0.5 * width, results['ACO'], width, label='ACO')
# plt.bar(x + 0.5 * width, results['Firefly'], width, label='Firefly')
plt.bar(x + 0.5 * width, results['Genetic'], width, label='Genetic', color = 'lightblue')

plt.xlabel('Number of Tasks')
plt.ylabel('Latency (seconds)')
plt.title('Latency Comparison of Optimization Algorithms')
plt.xticks(x, dims)
plt.legend()
plt.tight_layout()
plt.show()
