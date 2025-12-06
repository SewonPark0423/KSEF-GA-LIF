
from multiprocessing import Pool
import random as py_random
import os

try:
    from brian2 import *
    from brian2 import prefs
    prefs.codegen.target = 'numpy'
    BRIAN2_AVAILABLE = True
except Exception:
    BRIAN2_AVAILABLE = False

import numpy as np

try:
    from microglia_simulator import compute_fitness, simulate_microglia
except Exception:
    compute_fitness = None
    simulate_microglia = None

try:
    from deap import creator, base, tools, algorithms
    DEAP_AVAILABLE = True
except Exception:
    DEAP_AVAILABLE = False


def decode_chromosome(chrom):
    return {
        'k_sense': float(chrom[0]),
        'k_prune': float(chrom[1]),
        'k_resolve': float(chrom[2]),
        'k_protect': float(chrom[3])
    }


def run_brian_sim(params, env='dev', duration=100.0, seed=None):
    if not BRIAN2_AVAILABLE:
        if simulate_microglia is None:
            raise RuntimeError('Brian2 not available and microglia_simulator.simulate_microglia not importable')
        chrom = [params['k_sense'], params['k_prune'], params['k_resolve'], params['k_protect']]
        return simulate_microglia(chrom, env, E=0.6, T=int(duration))

    try:
        start_scope()
        if seed is not None:
            seed = int(seed)
            np.random.seed(seed)
            try:
                from brian2 import seed as brian_seed
                brian_seed(seed)
            except Exception:
                pass

        T = int(duration)
        N_e = 80
        N_i = 20

        base_rate = 5.0
        rate_scale = max(0.0, float(params.get('k_sense', 0.5))) * 30.0
        re = (base_rate + rate_scale) * Hz
        ri = (base_rate + 0.5 * rate_scale) * Hz

        Pe = PoissonGroup(N_e, rates=re)
        Pi = PoissonGroup(N_i, rates=ri)
        Me = SpikeMonitor(Pe)
        Mi = SpikeMonitor(Pi)
        run(T * ms)
        spikes_per_ms = np.zeros(T, dtype=float)
        for sp_t in Me.t/ms:
            idx = int(sp_t)
            if 0 <= idx < T:
                spikes_per_ms[idx] += 1
        for sp_t in Mi.t/ms:
            idx = int(sp_t)
            if 0 <= idx < T:
                spikes_per_ms[idx] += 1
        total_neurons = float(N_e + N_i)
        LIF_Activity_Input = spikes_per_ms / total_neurons
        A_history = np.clip(LIF_Activity_Input, 0.0, 1.0)
        k_sense = params.get('k_sense', 0.5)
        k_prune = params.get('k_prune', 0.5)
        k_resolve = params.get('k_resolve', 0.5)
        k_protect = params.get('k_protect', 0.5)

        S = 0.7
        D = 0.0
        S_history = []
        overpruned_time = 0
        chronic_A_time = 0
        E_level = 0.6

        for t in range(T):
            A = float(A_history[t])
            if env == 'dev':
                S_form = 0.05 * (1.0 - S)
            else:
                S_form = 0.0

            prune = k_prune * A * (1.0 - k_protect)
            S = S + S_form - prune
            S = float(max(0.0, min(1.0, S)))

            D = D + 0.02 * E_level - 0.03 * A
            D = float(max(0.0, D))

            S_history.append(S)
            if S < 0.3:
                overpruned_time += 1
            if A > 0.7:
                chronic_A_time += 1

        stats = {
            'S_history': np.array(S_history),
            'A_history': np.array(A_history),
            'S_final': float(S_history[-1]) if len(S_history) > 0 else 0.0,
            'S_mean': float(np.mean(S_history)) if len(S_history) > 0 else 0.0,
            'A_mean': float(np.mean(A_history)) if len(A_history) > 0 else 0.0,
            'D_final': float(D),
            'overpruned_time': int(overpruned_time),
            'chronic_A_time': int(chronic_A_time),
            'T': int(len(S_history)),
            'LIF_Input_History': np.array(LIF_Activity_Input)
        }

        return stats

    except Exception as e:
        print('Warning: Brian2 simulation failed with error:', str(e))
        if simulate_microglia is not None:
            chrom = [params['k_sense'], params['k_prune'], params['k_resolve'], params['k_protect']]
            try:
                return simulate_microglia(chrom, env, E=0.6, T=int(duration))
            except Exception as e2:
                print('Warning: simulate_microglia also failed:', str(e2))

        T = int(duration)
        rng = np.random.RandomState(seed if seed is not None else None)
        base_rate = 5.0
        rate_scale = max(0.0, float(params.get('k_sense', 0.5))) * 30.0
        lam = base_rate + rate_scale
        spikes_per_ms = rng.poisson(lam, size=T)
        total_neurons = 100.0
        A_history = np.clip(spikes_per_ms / total_neurons, 0.0, 1.0)

        k_prune = params.get('k_prune', 0.5)
        k_protect = params.get('k_protect', 0.5)
        S = 0.7
        S_history = []
        overpruned_time = 0
        chronic_A_time = 0
        D = 0.0
        E_level = 0.6
        for t in range(T):
            A = float(A_history[t])
            if env == 'dev':
                S_form = 0.05 * (1.0 - S)
            else:
                S_form = 0.0
            prune = k_prune * A * (1.0 - k_protect)
            S = S + S_form - prune
            S = float(max(0.0, min(1.0, S)))
            D = D + 0.02 * E_level - 0.03 * A
            D = float(max(0.0, D))
            S_history.append(S)
            if S < 0.3:
                overpruned_time += 1
            if A > 0.7:
                chronic_A_time += 1

        stats = {
            'S_history': np.array(S_history),
            'A_history': np.array(A_history),
            'S_final': float(S_history[-1]) if len(S_history) > 0 else 0.0,
            'S_mean': float(np.mean(S_history)) if len(S_history) > 0 else 0.0,
            'A_mean': float(np.mean(A_history)) if len(A_history) > 0 else 0.0,
            'D_final': float(D),
            'overpruned_time': int(overpruned_time),
            'chronic_A_time': int(chronic_A_time),
            'T': int(len(S_history)),
            'LIF_Input_History': np.array(A_history)
        }
        return stats


def evaluate_chromosome(chrom, env='dev', seed=None):
    params = decode_chromosome(chrom)
    stats = run_brian_sim(params, env=env, duration=100, seed=seed)
    if compute_fitness is None:
        raise RuntimeError('compute_fitness not available; ensure microglia_simulator.py is importable')
    fitness, objectives = compute_fitness(stats, env)
    return fitness, objectives


def parallel_evaluate_population(population, env='dev', n_workers=4):
    with Pool(processes=n_workers) as pool:
        args = [(list(ind), env, None) for ind in population]
        results = pool.starmap(evaluate_chromosome, args)
    fitness = np.array([r[0] for r in results])
    objs = [r[1] for r in results]
    return fitness, objs


def simple_ga_driver(env='dev', generations=50, pop_size=50, n_workers=4):
    from copy import deepcopy
    population = [list(np.random.rand(4)) for _ in range(pop_size)]

    for gen in range(generations):
        fitness, objs = parallel_evaluate_population(population, env=env, n_workers=n_workers)
        best_idx = int(np.argmax(fitness))
        print(f"[{env}] Gen {gen+1}/{generations} best fitness {fitness[best_idx]:.4f}")

        elites = [population[i] for i in np.argsort(fitness)[-max(2,int(0.1*pop_size)):]]
        new_pop = deepcopy(elites)
        while len(new_pop) < pop_size:
            p1, p2 = py_random.sample(elites, 2)
            child = [(a+b)/2.0 for a,b in zip(p1,p2)]
            for i in range(len(child)):
                if py_random.random() < 0.1:
                    child[i] += py_random.gauss(0, 0.05)
                    child[i] = max(0.0, min(1.0, child[i]))
            new_pop.append(child)
        population = new_pop

    fitness, objs = parallel_evaluate_population(population, env=env, n_workers=n_workers)
    best_idx = int(np.argmax(fitness))
    return population[best_idx], fitness, objs


def nsga2_driver(env='dev', generations=50, pop_size=50, n_workers=4, force_deap=False, deap_debug=False):
    if not DEAP_AVAILABLE:
        msg = 'DEAP not available; falling back to simple GA driver'
        if force_deap:
            raise RuntimeError(msg + ' (force_deap=True)')
        print(msg)
        best_ind, fitness, objs = simple_ga_driver(env=env, generations=generations, pop_size=pop_size, n_workers=n_workers)
        if compute_fitness is not None:
            stats = run_brian_sim({'k_sense': best_ind[0], 'k_prune': best_ind[1], 'k_resolve': best_ind[2], 'k_protect': best_ind[3]}, env=env, duration=100, seed=None)
            _, objd = compute_fitness(stats, env, multi_objective=True)
            S_final = objd.get('S_final', 0.0)
            if env == 'dev':
                penalty = objd.get('A_mean', 0.0) + objd.get('overpruned_frac', 0.0)
            else:
                penalty = objd.get('D_final', 0.0) + objd.get('chronic_A_frac', 0.0)
            pareto = [best_ind]
            pareto_objs = [(float(S_final), float(-penalty))]
            return pareto, pareto_objs

    import traceback
    fitness_cls = getattr(creator, 'FitnessMulti', None)
    individual_cls = getattr(creator, 'Individual', None)
    create_failed = False
    if fitness_cls is None:
        try:
            creator.create('FitnessMulti', base.Fitness, weights=(1.0, 1.0))
            fitness_cls = getattr(creator, 'FitnessMulti')
        except Exception:
            create_failed = True
            if deap_debug:
                print('Could not create creator.FitnessMulti; will try a unique name')
                traceback.print_exc()
    if individual_cls is None and fitness_cls is not None:
        try:
            creator.create('Individual', list, fitness=fitness_cls)
            individual_cls = getattr(creator, 'Individual')
        except Exception:
            create_failed = True
            if deap_debug:
                print('Could not create creator.Individual; will try a unique name')
                traceback.print_exc()

    if create_failed or fitness_cls is None or individual_cls is None:
        import time, random
        suffix = f'auto_{int(time.time())}_{random.randint(0,9999)}'
        fname = f'FitnessMulti_{suffix}'
        iname = f'Individual_{suffix}'
        try:
            creator.create(fname, base.Fitness, weights=(1.0, 1.0))
            fitness_cls = getattr(creator, fname)
            if deap_debug:
                print(f'Created fitness class {fname}')
        except Exception:
            if deap_debug:
                print('Unique-name Fitness creation failed:')
                traceback.print_exc()

        try:
            creator.create(iname, list, fitness=fitness_cls)
            individual_cls = getattr(creator, iname)
            if deap_debug:
                print(f'Created individual class {iname}')
        except Exception:
            if deap_debug:
                print('Unique-name Individual creation failed:')
                traceback.print_exc()

    if individual_cls is None or fitness_cls is None:
        msg = 'DEAP present but creator not fully initialized; falling back to simple GA driver'
        if force_deap:
            raise RuntimeError(msg + ' (force_deap=True)')
        print(msg)
        best_ind, fitness_arr, objs_arr = simple_ga_driver(env=env, generations=generations, pop_size=pop_size, n_workers=n_workers)
        if compute_fitness is not None:
            stats = run_brian_sim({'k_sense': best_ind[0], 'k_prune': best_ind[1], 'k_resolve': best_ind[2], 'k_protect': best_ind[3]}, env=env, duration=100, seed=None)
            _, objd = compute_fitness(stats, env, multi_objective=True)
            S_final = objd.get('S_final', 0.0)
            if env == 'dev':
                penalty = objd.get('A_mean', 0.0) + objd.get('overpruned_frac', 0.0)
            else:
                penalty = objd.get('D_final', 0.0) + objd.get('chronic_A_frac', 0.0)
            pareto = [best_ind]
            pareto_objs = [(float(S_final), float(-penalty))]
            return pareto, pareto_objs
        else:
            return [best_ind], [(0.0, 0.0)]

    toolbox = base.Toolbox()
    toolbox.register('attr_float', py_random.random)
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_float, n=4)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    def evaluate_individual(individual):
        params = list(individual)
        stats = run_brian_sim({'k_sense': params[0], 'k_prune': params[1], 'k_resolve': params[2], 'k_protect': params[3]}, env=env, duration=100, seed=None)
        if compute_fitness is None:
            raise RuntimeError('compute_fitness not available; ensure microglia_simulator.py is importable')
        _, objs = compute_fitness(stats, env, multi_objective=True)
        S_final = objs.get('S_final', 0.0)
        if env == 'dev':
            penalty = objs.get('A_mean', 0.0) + objs.get('overpruned_frac', 0.0)
        else:
            penalty = objs.get('D_final', 0.0) + objs.get('chronic_A_frac', 0.0)
        return (float(S_final), float(-penalty))

    toolbox.register('evaluate', evaluate_individual)
    toolbox.register('mate', tools.cxSimulatedBinaryBounded, low=0.0, up=1.0, eta=20.0)
    toolbox.register('mutate', tools.mutPolynomialBounded, low=0.0, up=1.0, eta=20.0, indpb=1.0/4.0)
    toolbox.register('select', tools.selNSGA2)

    pop = toolbox.population(n=pop_size)

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    try:
        pop = toolbox.select(pop, len(pop))
    except Exception:
        pass

    for gen in range(1, generations+1):
        k = len(pop)
        if k % 4 != 0:
            k = k - (k % 4)
            if k < 4:
                k = 4
        offspring = tools.selTournamentDCD(pop, k)
        if len(offspring) < len(pop):
            import random
            while len(offspring) < len(pop):
                offspring.append(toolbox.clone(random.choice(offspring)))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if py_random.random() < 0.9:
                toolbox.mate(ind1, ind2)
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
        if invalid_inds:
            results = list(map(toolbox.evaluate, invalid_inds))
            for ind, fit in zip(invalid_inds, results):
                ind.fitness.values = fit

        pop = toolbox.select(pop + offspring, pop_size)

        best = tools.selBest(pop, 1)[0]
        print(f'[NSGA-II {env}] Gen {gen}/{generations} best objectives {best.fitness.values}')

    pareto = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
    pareto_objs = [ind.fitness.values for ind in pareto]
    return pareto, pareto_objs


if __name__ == '__main__':
    best, fitness, objs = simple_ga_driver(env='dev', generations=3, pop_size=12, n_workers=2)
    print('Best:', best)
