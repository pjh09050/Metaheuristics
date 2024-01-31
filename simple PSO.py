import random
import numpy as np
from matplotlib import pyplot as plt
plt.rc('font', family='Malgun Gothic')
import matplotlib
import sys
import time
matplotlib.rcParams['animation.embed_limit'] = 50.0

class PSO:
    def __init__(self, fitness_function, bounds, num_particles, max_iter):
        self.fitness_func = fitness_function
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.global_best_position = []  
        self.global_best_fitness = -1  
        self.swarm = [] 
        for i in range(num_particles):
            self.swarm.append(self.create_particle())

    def create_particle(self):
        particle = {
            'position': [random.uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))],
            'velocity': [random.uniform(-1, 1) for _ in range(len(bounds))],
            'best_position': [],
            'fitness': sys.maxsize,
            'best_fitness': sys.maxsize,
            'iteration': 0
        }
        return particle

    def evaluate_fitness(self, particle):
        particle['fitness'] = self.fitness_func(particle['position'])
        if particle['fitness'] < particle['best_fitness']:
            particle['best_position'] = particle['position']
            particle['best_fitness'] = particle['fitness']

    def update_velocity(self, particle):
        w_min = 0.5
        w_max = 1
        particle['iteration'] += 1
        w = w_max - ((w_max - w_min) * particle['iteration'] / self.max_iter)
        c1 = 1
        c2 = 2
        for i in range(len(particle['position'])):
            r1 = random.random()
            r2 = random.random()
            cognitive_velocity = c1 * r1 * (particle['best_position'][i] - particle['position'][i])
            social_velocity = c2 * r2 * (self.global_best_position[i] - particle['position'][i])
            particle['velocity'][i] = w * particle['velocity'][i] + cognitive_velocity + social_velocity

    def update_position(self, particle):
        for i in range(len(particle['position'])):
            particle['position'][i] = particle['position'][i] + particle['velocity'][i]
            if particle['position'][i] < self.bounds[i][0]:
                particle['position'][i] = self.bounds[i][0]
            elif particle['position'][i] > self.bounds[i][1]:
                particle['position'][i] = self.bounds[i][1]

    def run_result(self):
        for i in range(self.max_iter):
            for particle in self.swarm:
                self.evaluate_fitness(particle)
                if particle['fitness'] < self.global_best_fitness or self.global_best_fitness == -1:
                    self.global_best_position = list(particle['position'])
                    self.global_best_fitness = float(particle['fitness'])
            for particle in self.swarm:
                self.update_velocity(particle)
                self.update_position(particle)

            if i % 10 == 0:
                print(f'Iteration {i}, Best Position: {self.global_best_position}, Fitness: {self.global_best_fitness:.6f}')

        print('Best position:', self.global_best_position)
        print('Best fitness:', self.global_best_fitness)
        return self.global_best_position, self.global_best_fitness

def objective_function(x):
    x = np.array(x)
    y = np.array(x)
    z = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0]**2 + y[1]**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * y[1]))) + np.exp(1) + 20
    return z


if __name__ == "__main__":
    bounds = [(-4,4), (-4,4)]  # 제약조건
    num_particles = 50
    maxiter = 100
    start_time = time.time()
    pso = PSO(objective_function, bounds, num_particles, maxiter)
    pso.run_result()