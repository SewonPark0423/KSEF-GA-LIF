import ga_optimizer
import matplotlib.pyplot as plt
import numpy as np


def plot_pareto_front(pareto_objs, env_name):
    x_data = [obj[1] for obj in pareto_objs]
    y_data = [obj[0] for obj in pareto_objs]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x_data, y_data, s=50, c='darkred', alpha=0.7, edgecolors='k', linewidths=0.5)
    
    sorted_points = sorted(zip(x_data, y_data), key=lambda x: x[0])
    sorted_x = [p[0] for p in sorted_points]
    sorted_y = [p[1] for p in sorted_points]
    
    ax.plot(sorted_x, sorted_y, color='darkred', linestyle='--', alpha=0.5, label='Pareto Front')
    ax.set_xlabel("Activation/Damage Penalty (Lower is Better)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Final Synaptic Health (S_final) (Higher is Better)", fontsize=12, fontweight='bold')
    ax.set_title(f"Pareto Front: Trade-off in {env_name.capitalize()} Environment", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_lif_microglia_correlation(stats, phenotype):
    time_steps = np.arange(len(stats['A_history']))
    lif_input = stats.get('LIF_Input_History', stats['A_history'])
    microglia_A = stats['A_history']

    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = '#1f77b4'
    ax1.set_xlabel('Time Step (t)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('LIF Input Activity (E)', color=color, fontsize=12, fontweight='bold')
    ax1.plot(time_steps, lif_input, color=color, label='LIF Input (E)')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = '#d62728'
    ax2.set_ylabel('Microglia Activation (A)', color=color, fontsize=12, fontweight='bold')
    ax2.plot(time_steps, microglia_A, color=color, linestyle='-', alpha=0.7, label='Microglia A(t)')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.axhline(y=0.7, color='grey', linestyle=':', alpha=0.6, label='Chronic A Threshold (0.7)')
    
    plt.title(f'LIF Input vs. Microglia Response (k_sense={phenotype[0]:.2f})', fontsize=14, fontweight='bold')
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("="*60)
    print("NSGA-II Multi-Objective Optimization")
    print("="*60)
    
    pareto, pareto_objs = ga_optimizer.nsga2_driver(env='dev', generations=10, pop_size=20, n_workers=2)
    print('\nPareto size:', len(pareto))
    print('Sample objs:', pareto_objs[:5])
    
    print("\n" + "="*60)
    print("Generating Scientific Analysis Plots...")
    print("="*60)
    
    plot_pareto_front(pareto_objs, 'Development')
    
    if pareto:
        best_ind = max(pareto, key=lambda ind: ind.fitness.values[0])
        best_params = ga_optimizer.decode_chromosome(best_ind)
        stats = ga_optimizer.run_brian_sim(best_params, env='dev')
        plot_lif_microglia_correlation(stats, best_ind)
    
    print("\n" + "="*60)
    print("Running Traditional GA from microglia_simulator.py")
    print("="*60)
    
    from microglia_simulator import run_ga, simulate_microglia
    
    result_dev = run_ga(env="dev", generations=30, pop_size=50)
    result_age = run_ga(env="aging", generations=30, pop_size=50)

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
    
    print("\n" + "="*60)
    print("All plots complete!")
    print("="*60)
