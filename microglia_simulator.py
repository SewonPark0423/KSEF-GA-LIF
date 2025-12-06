import numpy as np
import matplotlib.pyplot as plt
import random

TIME_STEPS = 100
POP_SIZE = 100
GENERATIONS = 60
ELITE_RATIO = 0.2
MUTATION_RATE = 0.1
MUTATION_STD = 0.1
INFLAMMATION_LEVEL = 0.6

def clip_param(x):
    return max(0.0, min(1.0, x))

def simulate_microglia(phenotype, env, E=INFLAMMATION_LEVEL, T=TIME_STEPS):
    k_sense, k_prune, k_resolve, k_protect = phenotype

    S = 0.7
    A = 0.0
    D = 0.0

    S_history = []
    A_history = []

    overpruned_time = 0
    chronic_A_time = 0

    for t in range(T):
        A = A + k_sense * E - k_resolve * A
        A = clip_param(A)

        if env == "dev":
            S_form = 0.05 * (1.0 - S)
        else:
            S_form = 0.0

        prune = k_prune * A * (1.0 - k_protect)
        S = S + S_form - prune
        S = clip_param(S)

        D = D + 0.02 * E - 0.03 * A
        D = max(0.0, D)

        S_history.append(S)
        A_history.append(A)

        if S < 0.3:
            overpruned_time += 1
        if A > 0.7:
            chronic_A_time += 1

    S_final = S_history[-1]
    S_mean = np.mean(S_history)
    A_mean = np.mean(A_history)

    # Return simulation statistics. Fitness calculation moved to compute_fitness()
    stats = {
        'S_history': np.array(S_history),
        'A_history': np.array(A_history),
        'S_final': float(S_final),
        'S_mean': float(S_mean),
        'A_mean': float(A_mean),
        'D_final': float(D),
        'overpruned_time': int(overpruned_time),
        'chronic_A_time': int(chronic_A_time),
        'T': int(T),
        'LIF_Input_History': np.array(A_history)
    }

    return stats

def initialize_population(pop_size=POP_SIZE):
    population = []
    for _ in range(pop_size):
        k_sense = random.random()
        k_prune = random.random()
        k_resolve = random.random()
        k_protect = random.random()
        population.append([k_sense, k_prune, k_resolve, k_protect])
    return np.array(population)

def evaluate_population(population, env, repeats=1, seeds=None):
    """Evaluate a population and return an array of fitness values.

    Args:
        population: array-like of individuals (chromosomes)
        env: 'dev' or 'aging'
        repeats: number of repeated evaluations per individual (for noisy sims)
        seeds: optional list/iterable of seeds (length >= repeats) to use per repeat

    Returns:
        numpy array of mean fitness values (averaged over repeats)
    """
    fitness_list = []
    for ind in population:
        fit_vals = []
        for r in range(repeats):
            # If seeds provided, set RNGs for reproducibility per repeat
            if seeds is not None:
                try:
                    s = seeds[r]
                except Exception:
                    s = None
                if s is not None:
                    random.seed(s)
                    np.random.seed(int(s) & 0xFFFFFFFF)

            stats = simulate_microglia(ind, env)
            fitness, _ = compute_fitness(stats, env)
            fit_vals.append(fitness)
        fitness_list.append(float(np.mean(fit_vals)))
    return np.array(fitness_list)


def compute_fitness(stats, env, weights=None, multi_objective=False):
    """Compute fitness from simulation stats.

    stats: dict returned by simulate_microglia
    env: 'dev' or 'aging'
    weights: optional dict to override default weights
    multi_objective: if True, return (None, objectives_dict)
    returns: (fitness_value, objectives_dict)
    """
    S_final = stats['S_final']
    S_mean = stats['S_mean']
    A_mean = stats['A_mean']
    D_final = stats['D_final']
    overpruned_frac = stats['overpruned_time'] / max(1, stats['T'])
    chronic_A_frac = stats['chronic_A_time'] / max(1, stats['T'])

    # Default weights (as used originally)
    if weights is None:
        if env == 'dev':
            weights = {'w_Sf': 1.5, 'w_Sm': 1.0, 'w_overpruned': 0.5, 'w_A': 0.3}
        else:
            weights = {'w_Sf': 1.2, 'w_Sm': 0.8, 'w_D': 0.5, 'w_chronicA': 0.4}

    objectives = {}
    if env == 'dev':
        objectives['S_final'] = S_final
        objectives['S_mean'] = S_mean
        objectives['overpruned_frac'] = overpruned_frac
        objectives['A_mean'] = A_mean
        if multi_objective:
            return None, objectives

        fitness = (
            weights['w_Sf'] * S_final +
            weights['w_Sm'] * S_mean -
            weights['w_overpruned'] * overpruned_frac -
            weights['w_A'] * A_mean
        )
    elif env == 'aging':
        objectives['S_final'] = S_final
        objectives['S_mean'] = S_mean
        objectives['D_final'] = D_final
        objectives['chronic_A_frac'] = chronic_A_frac
        if multi_objective:
            return None, objectives

        fitness = (
            weights['w_Sf'] * S_final +
            weights['w_Sm'] * S_mean -
            weights['w_D'] * D_final -
            weights['w_chronicA'] * chronic_A_frac
        )
    else:
        objectives = {'S_final': S_final, 'S_mean': S_mean}
        fitness = 0.0

    return float(fitness), objectives

def select_elites(population, fitness_list, elite_ratio=ELITE_RATIO):
    n_elite = max(2, int(len(population) * elite_ratio))
    idx_sorted = np.argsort(fitness_list)[::-1]
    elites = population[idx_sorted[:n_elite]]
    return elites

def crossover(parent1, parent2):
    child = []
    for p1, p2 in zip(parent1, parent2):
        r = random.random()
        if r < 0.4:
            child.append(p1)
        elif r < 0.8:
            child.append(p2)
        else:
            child.append((p1 + p2) / 2.0)
    return np.array(child)

def mutate(individual, mutation_rate=MUTATION_RATE, mutation_std=MUTATION_STD):
    new_ind = individual.copy()
    for i in range(len(new_ind)):
        if random.random() < mutation_rate:
            new_ind[i] += random.gauss(0.0, mutation_std)
            new_ind[i] = clip_param(new_ind[i])
    return new_ind

def run_ga(env, generations=GENERATIONS, pop_size=POP_SIZE):
    population = initialize_population(pop_size)
    best_fitness_history = []
    mean_params_history = []

    for gen in range(generations):
        fitness_list = evaluate_population(population, env)
        best_fitness = np.max(fitness_list)
        best_fitness_history.append(best_fitness)

        mean_params = np.mean(population, axis=0)
        mean_params_history.append(mean_params)

        print(f"[{env}] Generation {gen+1}/{generations}, best fitness = {best_fitness:.3f}")

        elites = select_elites(population, fitness_list)

        new_population = []
        for e in elites:
            new_population.append(e)

        while len(new_population) < pop_size:
            parents = random.sample(list(elites), 2)
            child = crossover(parents[0], parents[1])
            child = mutate(child)
            new_population.append(child)

        population = np.array(new_population)

    final_fitness = evaluate_population(population, env)
    best_idx = np.argmax(final_fitness)
    best_individual = population[best_idx]

    return {
        "best_individual": best_individual,
        "best_fitness_history": np.array(best_fitness_history),
        "mean_params_history": np.array(mean_params_history),
        "final_population": population,
    }

if __name__ == "__main__":
    result_dev = run_ga(env="dev")
    result_age = run_ga(env="aging")

    print("\nOptimal phenotype evolved in Development (k_sense, k_prune, k_resolve, k_protect):")
    print(result_dev["best_individual"])

    print("\nOptimal phenotype evolved in Aging (k_sense, k_prune, k_resolve, k_protect):")
    print(result_age["best_individual"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(result_dev["best_fitness_history"], linewidth=2.5, color='#2E86AB', marker='o', markersize=3, label="Development", alpha=0.8)
    ax.plot(result_age["best_fitness_history"], linewidth=2.5, color='#A23B72', marker='s', markersize=3, label="Aging", alpha=0.8)
    ax.set_xlabel("Generation", fontsize=12, fontweight='bold')
    ax.set_ylabel("Best Fitness Score", fontsize=12, fontweight='bold')
    ax.set_title("Fitness Evolution Across Generations", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='lower right')
    plt.tight_layout()
    plt.show()

    labels = ["k_sense", "k_prune", "k_resolve", "k_protect"]
    colors = ['#F18F01', "#FF3300", '#6A994E', "#FA9092"]
    
    fig, ax = plt.subplots(figsize=(11, 6))
    for i in range(4):
        ax.plot(result_dev["mean_params_history"][:, i], linewidth=2, color=colors[i], label=f"{labels[i]}", alpha=0.85, marker='')
    ax.set_xlabel("Generation", fontsize=12, fontweight='bold')
    ax.set_ylabel("Parameter Value", fontsize=12, fontweight='bold')
    ax.set_title("Parameter Adaptation in Developmental Environment", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='best', ncol=2)
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(11, 6))
    for i in range(4):
        ax.plot(result_age["mean_params_history"][:, i], linewidth=2, color=colors[i], label=f"{labels[i]}", alpha=0.85, marker='')
    ax.set_xlabel("Generation", fontsize=12, fontweight='bold')
    ax.set_ylabel("Parameter Value", fontsize=12, fontweight='bold')
    ax.set_title("Parameter Adaptation in Aging Environment", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='best', ncol=2)
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.show()

    best_dev = result_dev["best_individual"]
    best_age = result_age["best_individual"]

    stats_dev = simulate_microglia(best_dev, env="dev")
    stats_age = simulate_microglia(best_age, env="aging")
    S_dev = stats_dev['S_history']
    A_dev = stats_dev['A_history']
    S_age = stats_age['S_history']
    A_age = stats_age['A_history']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(S_dev, linewidth=2.5, color="#00FFE1", label="Development", alpha=0.8)
    ax1.plot(S_age, linewidth=2.5, color='#D62828', label="Aging", alpha=0.8)
    ax1.fill_between(range(len(S_dev)), S_dev, alpha=0.2, color="#00FFE1")
    ax1.fill_between(range(len(S_age)), S_age, alpha=0.2, color='#D62828')
    ax1.set_xlabel("Time Step", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Synaptic Health (S)", fontsize=11, fontweight='bold')
    ax1.set_title("Synaptic Health Trajectory", fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    ax2.plot(A_dev, linewidth=2.5, color='#FF006E', label="Development", alpha=0.8)
    ax2.plot(A_age, linewidth=2.5, color="#FFBB00", label="Aging", alpha=0.8)
    ax2.fill_between(range(len(A_dev)), A_dev, alpha=0.2, color='#FF006E')
    ax2.fill_between(range(len(A_age)), A_age, alpha=0.2, color="#FFBB00")
    ax2.set_xlabel("Time Step", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Microglia Activation (A)", fontsize=11, fontweight='bold')
    ax2.set_title("Microglia Activation Trajectory", fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()