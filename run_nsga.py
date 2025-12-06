import ga_brian2

if __name__ == '__main__':
    pareto, pareto_objs = ga_brian2.nsga2_driver(env='dev', generations=3, pop_size=12, n_workers=2)
    print('Pareto size:', len(pareto))
    print('Sample objs:', pareto_objs[:5])
