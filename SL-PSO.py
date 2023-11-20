import numpy as np 
import random
import time

class SocialLearningPSO:
    def __init__(self, x_init, M, x_dim, max_iter, fitness_fun):
        self.x = x_init.copy()
        self.fitness_fun, self.max_iter = fitness_fun, max_iter
        self.M, self.x_dim = self.x.shape[0], self.x.shape[1]
        assert(self.M == M and self.x_dim == x_dim) 
        self.delta_x = np.zeros(self.x.shape)
        self.best_fitness = []

    def __evaluate(self):
        self.fitness_val = [self.fitness_fun(self.x[i,:]) for i in range(self.M)]

    def __sorted_by_fitness(self):
        self.fitness_idx = list(zip(self.fitness_val, [i for i in range(self.M)]))
        self.fitness_idx = sorted(self.fitness_idx, key = lambda x: x[0], reverse = True)
        self.fitness_idx, self.fitness_val = list(zip(*self.fitness_idx))[1], list(zip(*self.fitness_idx))[0]
        self.x = self.x[self.fitness_idx,:]
        self.best_fitness.append(self.fitness_val[-1])

    def __get_lprob(self):
        self.learning_prob = [1 - (i-1)/self.M for i in range(1, self.M + 1)]
    
    def set_constrs(self, lb, ub):
        assert(lb.size == self.x_dim and ub.size == self.x_dim)
        self.lb, self.ub = lb, ub

    def run(self, flag = True):
        start_time = time.time()
        self.delta_x_next, self.x_next = self.delta_x.copy(), self.x.copy()

        for iter_times in range(self.max_iter):
            self.__evaluate()
            self.__sorted_by_fitness()
            self.__get_lprob()

            for i in range(self.M - 1):
                prob = random.random()
                if prob <= self.learning_prob[i]:
                    for j in range(self.x_dim):
                        k = random.randint(i + 1, self.M - 1)
                        r1, r2, r3 = random.random(), random.random(), random.random()
                        self.delta_x_next[i,j] = r1 * self.delta_x[i,j] + r2 * (self.x[k,j] - self.x[i,j])
                        self.x_next[i,j] = self.x[i,j] + self.delta_x_next[i,j]
        
            self.delta_x, self.x = self.delta_x_next.copy(), self.x_next.copy()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Elapsed Time:', elapsed_time)
        print('best position:', self.x[self.fitness_idx,:][-1])
        print('best fitness:', self.best_fitness[-1])

def objective_function(x):
    x = np.array(x)
    y = np.array(x)
    z = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0]**2 + y[1]**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * y[1]))) + np.exp(1) + 20
    return z

social_learning_pso = SocialLearningPSO(x_init = np.random.randn(30, 2), M = 30, x_dim = 2, max_iter = 1000, fitness_fun = objective_function)
social_learning_pso.run()