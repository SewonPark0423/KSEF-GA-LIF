
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from ga_optimizer import nsga2_driver, run_brian_sim


FIG_DIR = "figs"
os.makedirs(FIG_DIR, exist_ok=True)


def coerce_pareto_params(pareto):
    """Coerce various pareto element types to list-of-floats (params)."""
    params_list = []
    for p in pareto:
        if p is None:
            continue
        try:
            # try list() conversion (works for DEAP Individual and lists)
            arr = list(p)
            params_list.append([float(x) for x in arr])
        except Exception:
            # try to extract attribute (e.g., if it's already a numpy array)
            try:
                arr = list(np.asarray(p))
                params_list.append([float(x) for x in arr])
            except Exception:
                # fallback: skip
                continue
    return params_list


def coerce_objs(objs):
    """Normalize objective representations.

    Return a list of dicts with at least keys `S_final` and `penalty`.
    If input is tuple (S_final, -penalty) we convert accordingly.
    If input is dict-like we try to read the keys.
    """
    out = []
    for o in objs:
        if o is None:
            out.append({'S_final': 0.0, 'penalty': 0.0})
            continue
        if isinstance(o, dict):
            S = float(o.get('S_final', o.get('S_mean', 0.0)))
            # penalty guess
            penalty = float(o.get('A_mean', o.get('overpruned_frac', 0.0)))
            # prefer explicit penalty keys if present
            if 'penalty' in o:
                penalty = float(o['penalty'])
            out.append({'S_final': S, 'penalty': penalty})
            continue
        # tuple/list case
        try:
            # expecting (S_final, -penalty)
            Sf = float(o[0])
            second = float(o[1])
            penalty = -second
            out.append({'S_final': Sf, 'penalty': penalty})
            continue
        except Exception:
            out.append({'S_final': 0.0, 'penalty': 0.0})
    return out


def plot_pareto_front(objs, outpath=None):
    objs_norm = coerce_objs(objs)
    S_vals = [d['S_final'] for d in objs_norm]
    penalty = [d['penalty'] for d in objs_norm]

    plt.figure(figsize=(6,5))
    sns.scatterplot(x=penalty, y=S_vals, s=50)
    plt.xlabel('Penalty (lower better)')
    plt.ylabel('S_final (higher better)')
    plt.title('Pareto front objectives')
    plt.grid(True, alpha=0.3)
    if outpath:
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()
    else:
        plt.show()


def plot_param_vs_Sfinal(params_list, objs, param_idx=1, param_name='k_prune', outpath=None):
    objs_norm = coerce_objs(objs)
    S_vals = [d['S_final'] for d in objs_norm]
    params = [p[param_idx] for p in params_list]

    plt.figure(figsize=(6,5))
    sns.scatterplot(x=params, y=S_vals, s=50)
    sns.regplot(x=params, y=S_vals, scatter=False, lowess=True, color='orange')
    plt.xlabel(param_name)
    plt.ylabel('S_final')
    plt.title(f'{param_name} vs S_final (Pareto set)')
    plt.grid(True, alpha=0.3)
    if outpath:
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()
    else:
        plt.show()


def plot_lif_timeseries(stats, outpath=None):
    # stats expected to contain 'A_history' (input spikes normalized) and 'S_history'
    A = np.asarray(stats.get('A_history', []))
    S = np.asarray(stats.get('S_history', []))
    T = max(len(A), len(S))
    t = np.arange(T)

    fig, ax1 = plt.subplots(figsize=(8,4))
    ax1.plot(t, A[:T], color='C0', label='A_history (spike fraction)')
    ax1.set_xlabel('time (ms)')
    ax1.set_ylabel('Spike fraction (A)', color='C0')
    ax1.tick_params(axis='y', labelcolor='C0')

    ax2 = ax1.twinx()
    ax2.plot(t, S[:T], color='C1', label='S_history (microglia activity)')
    ax2.set_ylabel('S (microglia activity)', color='C1')
    ax2.tick_params(axis='y', labelcolor='C1')

    fig.legend(loc='upper right')
    plt.title('LIF activity (A) and Microglia activity (S) time-series')
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath)
        plt.close()
    else:
        plt.show()


def plot_params_boxplot(params_env_map, outpath=None):
    # params_env_map: dict env -> list of param-lists
    rows = []
    for env, params_list in params_env_map.items():
        for p in params_list:
            rows.append({'env': env, 'k_sense': p[0], 'k_prune': p[1], 'k_resolve': p[2], 'k_protect': p[3]})
    df = pd.DataFrame(rows)
    df_melt = df.melt(id_vars=['env'], value_vars=['k_sense','k_prune','k_resolve','k_protect'], var_name='param', value_name='value')

    plt.figure(figsize=(8,5))
    sns.boxplot(x='param', y='value', hue='env', data=df_melt)
    plt.title('Parameter distributions: dev vs aging (Pareto sets)')
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath)
        plt.close()
    else:
        plt.show()


def run_all_and_plot(generations=10, pop_size=40, n_workers=2):
    results = {}
    envs = ['dev', 'aging']
    for env in envs:
        print(f'Running NSGA-II (or fallback) for env={env} gens={generations} pop={pop_size}')
        pareto, pareto_objs = nsga2_driver(env=env, generations=generations, pop_size=pop_size, n_workers=n_workers)
        params_list = coerce_pareto_params(pareto)
        results[env] = {'pareto': pareto, 'pareto_params': params_list, 'pareto_objs': pareto_objs}

    # Plot Pareto front for dev (if available) using its objs
    if 'dev' in results:
        dev_objs = results['dev']['pareto_objs']
        plot_pareto_front(dev_objs, outpath=os.path.join(FIG_DIR, 'pareto_front.png'))

    # Plot k_prune vs S_final for dev
    if 'dev' in results and len(results['dev']['pareto_params'])>0:
        plot_param_vs_Sfinal(results['dev']['pareto_params'], results['dev']['pareto_objs'], param_idx=1, param_name='k_prune', outpath=os.path.join(FIG_DIR, 'kprune_vs_Sfinal.png'))

    # Time-series example: simulate a single pareto individual from dev (median)
    if 'dev' in results and len(results['dev']['pareto_params'])>0:
        idx = len(results['dev']['pareto_params'])//2
        chosen = results['dev']['pareto_params'][idx]
        params = {'k_sense': chosen[0], 'k_prune': chosen[1], 'k_resolve': chosen[2], 'k_protect': chosen[3]}
        stats = run_brian_sim(params, env='dev', duration=200, seed=42)
        plot_lif_timeseries(stats, outpath=os.path.join(FIG_DIR, 'lif_timeseries_example.png'))

    # Boxplot comparing dev vs aging parameter distributions
    params_env_map = {env: results[env]['pareto_params'] for env in envs if len(results[env]['pareto_params'])>0}
    if params_env_map:
        plot_params_boxplot(params_env_map, outpath=os.path.join(FIG_DIR, 'params_boxplot_dev_vs_aging.png'))

    print('Saved figures to', FIG_DIR)


if __name__ == '__main__':
    # small default run to keep runtime reasonable; increase as needed
    run_all_and_plot(generations=5, pop_size=20, n_workers=2)
