# imports
import itertools
import math
import os
import random
import numpy as np
import shutil
from scipy import special
from sys import argv

# global functions
def bernoulli(p): # random draw from bernoulli distribution
    return np.random.binomial(1, p)

class Simulation:
    def __init__(self, T, algorithm, parameters, trial):
        if not os.path.exists(f'results/{parameters}'): os.mkdir(f'results/{parameters}') # create a folder
        if not os.path.exists(f'results/{parameters}/{algorithm}'): os.mkdir(f'results/{parameters}/{algorithm}') # create a folder

        self.trial = trial
        self.algorithm = algorithm
        self.parameters = parameters

        self.min_width = True if algorithm == 'min-width' else False
        self.min_UCB_alg = True if algorithm == 'min-UCB' else False
        self.no_sharing_alg = True if algorithm == 'no-sharing' else False
        self.cucb_alg = True if algorithm == 'cucb' else False
        self.ucb = True if algorithm == 'ucb' else False
        self.random_alg = True if algorithm == 'random' else False
        
        self.delta = 0.05 # confidence parameter
        self.t = 1 # start time
        self.T = T # time horizon
        self.means = list(map(float, parameters.split('-')[0].split('=')[1].split(','))) # true means of the arms
        self.N = len(self.means) # number of arms
        self.profiles = list(map(float, parameters.split('-')[1].split('=')[1].split(','))) # sensitivities of the agents
        self.estimated_profiles = list(map(float, parameters.split('-')[2].split('=')[1].split(','))) # estimated sensitivities of the agents
        self.A = len(self.profiles) # number of agents
        self.action_record = np.zeros((0, self.A, self.N)) # action_record[t, a, n] = 1 if agent a visited arm n in time step t and 0 otherwise
        self.cumulative_record = np.zeros((0, self.A, self.N)) # cumulative_record[t, a, n] = number of times agent a has visited arm n through time step t
        self.observations = np.zeros((0, self.A, self.N)) # observations[t, a, n] = observation of agent a in arm n at time step t (1 or 0)

        self.agent_means = np.full((self.A, self.N), 0.5) # initialize agent means to 0.5
        self.agent_epsilons = np.full((self.A, self.N), np.inf) # initialize agent epsilons to infinity
        self.agent_UCBs = np.full((self.A, self.N), np.inf) # initialize agent UCBs to infinity

        self.shared_means = self.N * [0.5] # initialize shared means to 0.5
        self.shared_epsilons = self.N * [np.inf] # initialize shared epsilons to infinity
        self.shared_UCBs = self.N * [np.inf] # initialize shared UCBs to infinity
        
        if self.ucb:
            self.super_arms = np.array(list(map(list, itertools.permutations(range(self.N), self.A))))
            self.empirical_means = len(self.super_arms) * [0.5] # initialize shared means to 0.5
            self.epsilons = len(self.super_arms) * [np.inf] # initialize shared epsilons to infinity
            self.UCBs = len(self.super_arms) * [np.inf] # initialize shared UCBs to infinity
            self.superarm_action_record = np.zeros((0, len(self.super_arms))) # superarm_action_record[t, f] = 1 if super-arm f is pulled at time t and 0 otherwise
            self.superarm_cumulative_record = np.zeros((0, len(self.super_arms))) 
            self.superarm_observations = np.zeros((0, len(self.super_arms))) # superarm_observations[t, f] = reward of super-arm f at time t
        
        self.regret = [] # regret[t] = cumulative regret at time step t

        ranked_agents = np.flip(np.argsort(self.estimated_profiles)) # lists the agents in descending order according to sensitivity
        ranked_arms = np.flip(np.argsort(self.means)) # lists the arms in descending order according to mean
        self.optimal_configuration = [ranked_arms[np.where(ranked_agents == a)[0][0]] for a in range(self.A)] # the optimal configuration assigns the ith best agent to the ith best arm
        # print(f'Optimal configuration: {self.optimal_configuration}')

        while self.t <= self.T: # t = 1,...,T
            if self.ucb: configuration = self.super_arms[np.random.choice(np.where(np.array(self.UCBs) == max(self.UCBs))[0])]
            
            else:
                configuration = self.A * [-1] # configuration[a] is the arm to which agent a will be assigned
                unassigned_arms = list(range(self.N)) # initially none of the arms have been assigned an agent
                
                if self.cucb_alg: agents = random.sample(list(ranked_agents), len(ranked_agents))
                else: agents = ranked_agents

                for a in agents: # loops through the agents from best to worst
                    if self.min_width or self.min_UCB_alg or self.cucb_alg: UCBs = np.take(self.shared_UCBs, unassigned_arms)

                    elif self.no_sharing_alg: UCBs = np.take(self.agent_UCBs[a], unassigned_arms)

                    n = unassigned_arms[np.random.choice(np.where(UCBs == max(UCBs))[0])] # select the arm with the highest UCB out of those that have not been assigned an agent yet
                    configuration[a] = n # assign the agent to the selected arm
                    unassigned_arms.remove(n) # remove the assigned arm from the unassigned list

            self.take_action(configuration) # pull the arms to which the agents have been assigned 
            if self.t % 5000 == 0: np.save(f'results/{self.parameters}/{self.algorithm}/{self.algorithm}-cumulative-regret-{self.trial}', self.regret) # save cumulative regret
            self.t += 1      

    def take_action(self, configuration):
        configuration_matrix = np.zeros((self.A, self.N)) # each row has a single 1 with the remaining entries 0
        observation_matrix = np.zeros((self.A, self.N)) # each row may have a single 1 with the remaining entries 0

        for a in range(self.A): # for each agent
            n = configuration[a]
            configuration_matrix[a][n] = 1 # record the assignment
            observation_matrix[a][n] = bernoulli(self.profiles[a] * self.means[n]) # record the observation
        
        self.action_record = np.append(self.action_record, [configuration_matrix], axis = 0) # add the action to the history
        self.cumulative_record = np.append(self.cumulative_record, [np.sum(self.action_record, axis = 0)], axis = 0)
        self.observations = np.append(self.observations, [observation_matrix], axis = 0) # add the observation to the history

        if self.ucb:
            configuration_vector = np.zeros(len(self.super_arms))
            configuration_vector[np.where(np.all(self.super_arms == configuration, axis = 1))[0][0]] = 1

            observation_vector = np.zeros(len(self.super_arms))
            observation_vector[np.where(np.all(self.super_arms == configuration, axis = 1))[0][0]] = np.sum(observation_matrix)

            self.superarm_action_record = np.append(self.superarm_action_record, [configuration_vector], axis = 0)
            self.superarm_cumulative_record = np.append(self.superarm_cumulative_record, [np.sum(self.superarm_action_record, axis = 0)], axis = 0)
            self.superarm_observations = np.append(self.superarm_observations, [observation_vector], axis = 0) # add the observation to the history

            for f in range(len(self.super_arms)):
                self.empirical_means[f] = self.get_mean_estimate(f)
                self.epsilons[f] = self.get_epsilon(f)
                self.UCBs[f] = self.get_UCB(f)

        else:
            for n in range(self.N):
                if not self.min_width:
                    for a in range(self.A):
                        self.agent_means[a][n] = self.get_mean_estimate(n, a)
                        self.agent_epsilons[a][n] = self.get_epsilon(n, a)
                        self.agent_UCBs[a][n] = self.get_UCB(n, a)

                if not self.no_sharing_alg:
                    if not self.min_UCB_alg:
                        self.shared_means[n] = self.get_mean_estimate(n)
                        self.shared_epsilons[n] = self.get_epsilon(n)
                    
                    self.shared_UCBs[n] = self.get_UCB(n)
        
        self.regret.append(self.get_regret()) # record the cumulative regret

        # print('t =', self.t)
        # print('Action record =\n', self.action_record[-1])
        # # print('Super-arm action record =\n', self.superarm_action_record[-1]) 
        # # print('Cumulative record =\n', self.cumulative_record[-1])
        # print('Observations =\n', self.observations[-1])
        # # print('Super-arm observations =\n', self.superarm_observations[-1])
        # # print('Agent mean estimates =\n', np.around(self.agent_means, 3))
        # print('Shared mean estimates =\n', np.around(self.shared_means, 3))
        # # print('Empirical mean estimates =\n', np.around(self.empirical_means, 3))
        # # print('Agent epsilons =\n', np.around(self.agent_epsilons, 3))
        # # print('Shared epsilons =\n', np.around(self.shared_epsilons, 3))
        # # print('Epsilons =\n', np.around(self.epsilons, 3))
        # # print('Agent UCBs =\n', np.around(self.agent_UCBs, 3))
        # print('Shared UCBs =\n', np.around(self.shared_UCBs, 3))
        # # print('UCBs =\n', np.around(self.UCBs, 3))
        # print('Regret =', round(self.regret[-1], 2))
        
    def get_c_agent(self, n, a): # number of times arm n has been pulled by agent a after time step self.t
        return np.sum([self.action_record[tau, a, n] for tau in range(self.t)])
    
    def get_mean_estimate(self, n, a = None): # estimate for mean of arm n after time self.t
        if a == None:
            if self.ucb:
                if self.superarm_cumulative_record[-1, n] == 0: return 0.5
                return 1 / self.superarm_cumulative_record[-1, n] * np.sum([self.superarm_action_record[tau, n] * self.superarm_observations[tau, n] for tau in range(self.t)])
            
            if np.sum(self.cumulative_record[-1].T[n]) == 0: return 0.5
            if self.min_width: return 1 / np.sum([self.estimated_profiles[a] ** 2 * self.cumulative_record[-1, a, n] for a in range(self.A)]) * np.sum([self.estimated_profiles[a] * np.sum([self.action_record[tau, a, n] * self.observations[tau, a, n] for tau in range(self.t)]) for a in range(self.A)])
            if self.cucb_alg: return np.sum(np.sum(self.observations, axis = 0), axis = 0)[n] / np.sum(self.cumulative_record[-1], axis = 0)[n]

        if a != None:
            if self.cumulative_record[-1, a, n] == 0: return 0.5
            return 1 / (self.estimated_profiles[a] * self.cumulative_record[-1, a, n]) * np.sum([self.action_record[tau, a, n] * self.observations[tau, a, n] for tau in range(self.t)])
        
    def get_epsilon(self, n, a = None):
        if a == None:
            if self.ucb:
                if self.superarm_cumulative_record[-1, n] == 0: return np.inf
                return np.sqrt(np.log(2 * len(self.super_arms) * self.t / self.delta) / (2 * self.superarm_cumulative_record[-1, n]))

            if np.sum(self.cumulative_record[-1].T[n]) == 0: return np.inf

            if self.min_width:
                factor = np.sum([special.comb(i + self.A, self.A - 1) for i in range(self.t)])
                return np.sqrt(np.log(2 * self.N * factor / self.delta) / (2 * np.sum([self.estimated_profiles[a] ** 2 * self.cumulative_record[-1, a, n] for a in range(self.A)])))
                        
            if self.cucb_alg: return np.sqrt(np.log(2 * self.N * self.t / self.delta) / (2 * np.sum(self.cumulative_record[-1], axis = 0)[n]))
            
        if a != None:
            if self.cumulative_record[-1, a, n] == 0: return np.inf
            return 1 / self.estimated_profiles[a] * np.sqrt(np.log(2 * self.A * self.N * self.t / self.delta) / (2 * self.cumulative_record[-1, a, n]))

    def get_UCB(self, n, a = None):
        if a == None:
            if self.min_UCB_alg:
                return min(self.agent_UCBs.T[n])
            
            if self.ucb: return self.empirical_means[n] + self.epsilons[n]

            return self.shared_means[n] + self.shared_epsilons[n]
        
        if a != None: return self.agent_means[a][n] + self.agent_epsilons[a][n]

    def get_regret(self):
        return np.sum([np.sum([self.profiles[a] * (self.means[self.optimal_configuration[a]] - self.means[np.where(self.action_record[tau, a] == 1)[0][0]]) for a in range(self.A)]) for tau in range(self.t)])

algorithm = argv[1] # get algorithm specified by user
trial = int(argv[2]) # get trial specified by user
parameters = argv[3]
horizon = int(argv[4])

# print(f'{algorithm}, {trial}')
simulation = Simulation(T = horizon, algorithm = algorithm, parameters = parameters, trial = trial)
cumulative_regret = simulation.regret

np.save(f'results/{parameters}/{algorithm}/{algorithm}-cumulative-regret-{trial}', cumulative_regret) # save cumulative regret