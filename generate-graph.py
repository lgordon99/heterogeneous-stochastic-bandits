# imports
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

def graph(T, parameters, save, top, xticks, yticks, num_trials=90, bottom=0.65, right=0.7, regret_type='cumulative', with_ucb=True, fontsize=12, legend=True, ylabel=True, title=None):
    colors = ['b', 'g', 'r', 'c', 'm']
    labels = ['M-W', 'M-UCB', 'N-S', 'CUCB', 'UCB']
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (0, (1,10))]
    algorithms = ['min-width', 'min-UCB', 'no-sharing', 'cucb', 'ucb'] if with_ucb else ['min-width', 'min-UCB', 'no-sharing', 'cucb']
    regrets = np.empty((len(algorithms), num_trials, T))

    for i, algorithm in enumerate(algorithms):
        print(f'Loading {algorithm} data...')
        regrets[i] = np.array([np.load(f'results/{parameters}/{algorithm}/{array_name}')[:T].tolist() for array_name in os.listdir(f'results/{parameters}/{algorithm}') if len(np.load(f'results/{parameters}/{algorithm}/{array_name}')) >= T and int(array_name.split('-')[-1].split('.')[0]) < 500])[:num_trials]
        # if i > 0:
        #     regrets[i] = np.array([np.load(f'results/{parameters}/{algorithm}/{array_name}', allow_pickle=True)[:T].tolist() for array_name in os.listdir(f'results/{parameters}/{algorithm}') if len(np.load(f'results/{parameters}/{algorithm}/{array_name}', allow_pickle=True)) >= T])[:num_trials]
        # else:
        # regrets[i] = np.array([np.load(f'results/{parameters}/{algorithm}/{array_name}')[:T] for array_name in os.listdir(f'results/{parameters}/{algorithm}') if int(array_name.split('-')[-1].split('.')[0]) >= 500])[:num_trials]
    if save: np.save(f'results/saved-arrays/regrets_{parameters}', regrets)

    regret_means = np.mean(regrets, axis=1)
    regret_standard_errors = np.std(regrets.astype(np.float64), axis=1) / np.sqrt(num_trials)

    print(regrets.shape)
    print(regret_means.shape)
    print(regret_standard_errors.shape)

    fig, ax = plt.subplots(dpi=300)
    plt.subplots_adjust(bottom=bottom, right=right)

    for i in range(len(algorithms)):
        plt.plot(range(1, T+1), regret_means[i], c=colors[i], label=labels[i], linestyle=linestyles[i]) # plot regret
        plt.fill_between(range(1, T+1), regret_means[i] - 2*regret_standard_errors[i], regret_means[i] + 2*regret_standard_errors[i], color=colors[i], alpha=0.5, lw=0) # plot two standard errors

    # plt.xscale('log')
    plt.xticks(range(0, T+1, xticks), fontsize=fontsize)
    plt.xlabel('Time', fontsize=fontsize)
    plt.ylim(0, top)
    plt.yticks(range(0, top+1, yticks), fontsize=fontsize)
    if title is not None: plt.title(title)

    if ylabel:
        if regret_type == 'cumulative':
            plt.ylabel('Regret', fontsize=fontsize)
        else:
            plt.ylabel('Incremental Regret', fontsize=fontsize)

    if legend: plt.legend(fontsize=10, ncol=2, labelspacing=0.3, handlelength=2, columnspacing=0.5, handletextpad=0.5)
    plt.savefig(f'results/graphs/{parameters}-{regret_type}-regret.png', bbox_inches='tight')

# parameters = 'mu=0.1,0.8,0.9-p=0.5,0.5'
# parameters = 'mu=0.1,0.3,0.5,0.7,0.9-p=0.2,0.3' # poaching 2 agents
# parameters = 'mu=0.1,0.3,0.5,0.7,0.9-p=0.1,0.3' # new poaching 2 agents
# parameters = 'mu=0.1,0.3,0.5,0.7,0.9-p=0.1,0.2,0.3' # poaching 3 agents
# parameters = 'mu=0.1,0.3,0.5,0.7,0.9-p=0.1,0.2,0.3-ep=0.2,0.33,0.33' # poaching 3 agents estimated sensitivities
# parameters = 'mu=0.1,0.3,0.5,0.7,0.9-p=0.1,0.1,0.1,0.2,0.3' # poaching 5 agents
# parameters = 'mu=0.1,0.3,0.5,0.7,0.9-p=0.1,0.1,0.1,0.2,0.3-ep=0.2,0.2,0.2,0.33,0.33' # poaching 5 agents estimated sensitivities
# parameters = 'mu=0.72,0.74,0.93,0.61-p=0.3,0.5,0.7,0.9' # hotel
# parameters = 'mu=0.1,0.5-p=0.1,0.9'
# parameters = 'mu=0.1,0.8,0.9-p=0.1,0.9'
# parameters = 'mu=0.1,0.2,0.9-p=0.5,0.9'
# parameters = 'mu=0.1,0.9-p=0.9'
# parameters = 'mu=0.074,0.085,0.104,0.114,0.12,0.14-p=0.8,0.8,0.8,0.8,0.95' # covid
# parameters = 'mu=0.074,0.085,0.104,0.114,0.12,0.14-p=0.8,0.8,0.8,0.8,0.95-ep=0.9,0.9,0.9,0.9,0.98' # covid with estimated sensitivities
# parameters = 'mu=0.05,0.1,0.12,0.15,0.25,0.3-p=0.8,0.8,0.8,0.95,0.95' # new covid
# parameters = 'mu=0.05,0.1,0.12,0.15,0.25,0.3-p=0.8,0.8,0.8,0.95,0.95-ep=0.9,0.9,0.9,0.98,0.98' # new covid with estimated sensitivities
# parameters = 'mu=0.05,0.1,0.12,0.15,0.25,0.3-p=0.8,0.8,0.8,0.95,0.95-ep=0.75,0.75,0.75,0.98,0.98' # new covid with estimated sensitivities
# parameters = 'mu=0.05,0.1,0.12,0.15,0.25,0.3-p=0.8,0.8,0.8,0.95,0.95-ep=0.75,0.75,0.75,0.9,0.9' # new covid with estimated sensitivities
# parameters = 'mu=0.05,0.1,0.12,0.15,0.25,0.3-p=0.8,0.8,0.8,0.95,0.95-ep=0.95,0.95,0.95,0.98,0.98' # new covid with estimated sensitivities
parameters = 'mu=0.05,0.1,0.12,0.15,0.25,0.3-p=0.8,0.8,0.8,0.95,0.95-ep=0.85,0.85,0.85,0.98,0.98' # new covid with estimated sensitivities

# graph(T=45000, parameters=parameters, save=False, top=500, xticks=10000, yticks=100, num_trials=90, legend=True, ylabel=True, with_ucb=True) # covid long time
# graph(T=300, parameters=parameters, save=True, top=30, xticks=50, yticks=10, num_trials=500, legend=True, ylabel=True, with_ucb=True, title='(a) COVID test allocation') # covid short time
graph(T=300, parameters=parameters, save=True, top=30, xticks=50, yticks=10, num_trials=500, legend=True, ylabel=True, with_ucb=True) # covid short time

# graph(T=5000, parameters=parameters, save=False, top=200, xticks=1000, yticks=50, num_trials=90, legend=False, ylabel=False, title='(b) Hotel recommendation') # hotel

# graph(T=5000, parameters=parameters, save=False, top=200, xticks=1000, yticks=50, num_trials=90, legend=True, ylabel=True, with_ucb=True, title='(a) 2 agents: {0.2,0.3}') # poaching 2 agents short time
# graph(T=5000, parameters=parameters, save=True, top=300, xticks=1000, yticks=100, num_trials=90, legend=False, ylabel=False, title='(b) 3 agents: {0.1,0.2,0.3}') # poaching 3 agents short time
# graph(T=5000, parameters=parameters, save=True, top=200, xticks=1000, yticks=50, num_trials=90, legend=False, ylabel=False, title='(c) 5 agents: {0.1,0.1,0.1,0.2,0.3}') # poaching 5 agents short time

# graph(T=90000, parameters=parameters, save=False, top=1000, xticks=30000, yticks=200, num_trials=90, legend=False, ylabel=False) # poaching 2 agents
# graph(T=60000, parameters=parameters, save=False, top=1000, xticks=20000, yticks=200, num_trials=90, legend=False, ylabel=False) # poaching 3 agents
# graph(T=30000, parameters=parameters, save=True, top=1000, xticks=10000, yticks=200, num_trials=90, legend=False, ylabel=False) # poaching 5 agents
