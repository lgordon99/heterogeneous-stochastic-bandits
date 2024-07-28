# imports
import numpy as np
import os
import scipy.stats as stats

with_ucb = True
algorithms = ['min-width', 'min-UCB', 'no-sharing', 'cucb', 'ucb'] if with_ucb else ['min-width', 'min-UCB', 'no-sharing', 'cucb']
parameters = 'mu=0.05,0.1,0.12,0.15,0.25,0.3-p=0.8,0.8,0.8,0.95,0.95'

# estimated_sensitivities = '0.75,0.75,0.75,0.9,0.9'
estimated_sensitivities = '0.75,0.75,0.75,0.98,0.98'
# estimated_sensitivities = '0.85,0.85,0.85,0.98,0.98'

regrets = np.load(f'results/saved-arrays/regrets_{parameters}.npy')
num_trials = regrets.shape[1]
print(f'num_trials = {num_trials}')
print(f'Regrets shape = {regrets.shape}')
last_regrets = np.array([regrets[i].T[-1].T for i in range(len(regrets))])
print(f'Last regrets shape = {last_regrets.shape}')
regrets_est = np.load([f'results/saved-arrays/{file}' for file in os.listdir('results/saved-arrays') if parameters in file and estimated_sensitivities in file][0])
print(f'Regrets estimated sensitivities shape = {regrets_est.shape}')
last_regrets_est = np.array([regrets_est[i].T[-1].T for i in range(len(regrets_est))])
print(f'Last regrets est shape = {last_regrets_est.shape}')

# for i, algorithm in enumerate(algorithms):
#     print(f'{algorithm}: p={stats.ttest_ind(regrets[i].T[-1].T, regrets_est[i].T[-1].T)[1]}, regret worse by {np.mean(regrets_est[i].T[-1].T)-np.mean(regrets[i].T[-1].T)}')

percent_changes = (last_regrets_est - last_regrets) / last_regrets
print(f'Percent changes shape = {percent_changes.shape}')
percent_change_mean = np.mean(percent_changes, axis=1)
print(f'Percent change mean shape = {percent_change_mean.shape}')
percent_change_standard_error = np.std(percent_changes, axis=1) / np.sqrt(num_trials)
print(f'Percent change standard error shape = {percent_change_standard_error.shape}')

print(f'Last regrets est shape = {last_regrets_est.shape}')
last_regrets_est_mean = np.mean(last_regrets_est, axis=1)
print(f'Last regrets est mean shape = {last_regrets_est_mean.shape}')
last_regrets_est_standard_error = np.std(last_regrets_est, axis=1) / np.sqrt(num_trials)
print(f'Last regrets est standard error shape = {last_regrets_est_standard_error.shape}')

for i, algorithm in enumerate(algorithms):
    # print(f'{algorithm}: {round(100*percent_change_mean[i], 1)}% \u00B1 {round(100*percent_change_standard_error[i], 1)}%')
    print(f'{algorithm}: {round(last_regrets_est_mean[i], 1)} \u00B1 {round(last_regrets_est_standard_error[i], 1)}')
