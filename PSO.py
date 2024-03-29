import random
import numpy as np
from math import *
from matplotlib import pyplot as plt
plt.rc('font', family='Malgun Gothic')
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML
import matplotlib
import sys
import time
matplotlib.rcParams['animation.embed_limit'] = 50.0 # 용량 제한 

class Particle:
    def __init__(self, bounds, max_iter):
        self.position = []  # particle current position
        self.velocity = []  # particle current velocity
        self.best_position = [] # particle best position
        self.fitness = sys.maxsize   # particle fitness
        self.best_fitness = sys.maxsize  # particle best fitness
        self.iteration = 0 # 반복 횟수
        self.max_iter = max_iter
        for i in range(len(bounds)):
            self.position.append(random.uniform(bounds[i][0], bounds[i][1]))
            self.velocity.append(random.uniform(-1, 1))

    def evaluate_fitness(self, fitness_func):
        # current position에 대한 fitness 계산
        self.fitness = fitness_func(self.position)
        # best_position과 best_fitness 업데이트
        if self.fitness < self.best_fitness:
            self.best_position = self.position
            self.best_fitness = self.fitness

    def update_velocity(self, global_best_position):
        w_min = 0.5
        w_max = 1
        self.iteration += 1
        w = w_max - ((w_max - w_min) * self.iteration / self.max_iter) # w가 점점 감소
        #w = random.uniform(w_min, w_max) # particle의 속도에 대한 가중치 (w가 랜덤으로 설정)
        c1 = 1  # 자신의 최고 위치에 대한 가중치
        c2 = 2  # 집단의 최고 위치에 대한 가중치
        for i in range(len(self.position)):
            r1 = random.random() # 0,1 사이의 난수
            r2 = random.random()
            cognitive_velocity = c1 * r1 * (self.best_position[i] - self.position[i])
            social_velocity = c2 * r2 * (global_best_position[i] - self.position[i])
            self.velocity[i] = w * self.velocity[i] + cognitive_velocity + social_velocity

    def update_position(self, bounds):
        for i in range(len(self.position)):
            self.position[i] = self.position[i] + self.velocity[i]
            # adjust maximum position 
            if self.position[i] < bounds[i][0]:
                self.position[i] = bounds[i][0]
            # adjust minimum position 
            elif self.position[i] > bounds[i][1]:
                self.position[i] = bounds[i][1]

class PSO:
    '''
    animation : 처음부터 끝까지 (중간에 끊으면 안됨) 
    result : 값을 바로 알려줌
    '''
    def __init__(self, fitness_function, bounds, num_particles, max_iter):
        self.fitness_func = fitness_function
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.global_best_position = []  # group best position
        self.global_best_fitness = -1  # group best fitness
        self.swarm = [] # 초기 particles
        for i in range(num_particles):
            self.swarm.append(Particle(bounds, self.max_iter))
            
    def run_animation(self):
        history = [] 
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111, projection='3d')
        x = np.linspace(self.bounds[0][0], self.bounds[0][1], 100)
        y = np.linspace(self.bounds[1][0], self.bounds[1][1], 100)
        X, Y = np.meshgrid(x, y)
        Z = objective_function([X,Y])
        ax.plot_wireframe(X, Y, Z, color='r', linewidth=0.2)
        ax.set_xlim(self.bounds[0][0], self.bounds[0][1])
        ax.set_ylim(self.bounds[1][0], self.bounds[1][1])
        ax.set_zlim(0, 12)
        scatter = ax.scatter([], [], [], s=50, c='b', marker='o')

        def animate(i):
            history = []
            particle_history = []
            # 각 particle의 fitness 평가하기
            for j in range(self.num_particles):
                self.swarm[j].evaluate_fitness(self.fitness_func)
                # global_best_position, global_best_fitness 업데이트
                if self.swarm[j].fitness < self.global_best_fitness or self.global_best_fitness == -1:
                    self.global_best_position = list(self.swarm[j].position)
                    self.global_best_fitness = float(self.swarm[j].fitness)
                particle_history.append([self.swarm[j].position[0], self.swarm[j].position[1], self.swarm[j].fitness])
            # 각 particle의 position과 velocity 업데이트
            for j in range(self.num_particles):
                self.swarm[j].update_velocity(self.global_best_position)
                self.swarm[j].update_position(self.bounds)
            history.extend(particle_history)
            xs = [particle[0] for particle in history]
            ys = [particle[1] for particle in history]
            zs = [particle[2] for particle in history]
            scatter._offsets3d = (xs,ys,zs)
            #ax.set_title("PSO 해 변화 과정 (Iteration %d)" % i, fontsize=20)
            if i == self.max_iter-1:
                ax.scatter(self.global_best_position[0], self.global_best_position[1], self.global_best_fitness, c='r', marker='*', s=100)
                scatter.set_alpha(0)
            fig.canvas.draw()
            return scatter,
        
        anim = animation.FuncAnimation(fig, animate, frames=self.max_iter, interval=50, blit=False, repeat=False)
        
        plt.rcParams['axes.unicode_minus'] = False
        anim.save('PSO.gif', writer='pillow', fps=10) # gif저장
        plt.show() 
        print('Best position:', self.global_best_position)
        print('Best fitness:', self.global_best_fitness)      

        return self.global_best_position, self.global_best_fitness, history, HTML(anim.to_jshtml())
    
    def run_result(self):
        start_time = time.time()
        for i in range(self.max_iter):
            # 각 particle의 fitness 평가하기
            for j in range(self.num_particles):
                self.swarm[j].evaluate_fitness(self.fitness_func)
                # global_best_position, global_best_fitness 업데이트
                if self.swarm[j].fitness < self.global_best_fitness or self.global_best_fitness == -1:
                    self.global_best_position = list(self.swarm[j].position)
                    self.global_best_fitness = float(self.swarm[j].fitness)
            # 각 particle의 position과 velocity 업데이트
            for j in range(self.num_particles):
                self.swarm[j].update_velocity(self.global_best_position)
                self.swarm[j].update_position(self.bounds)

            if i % 10 == 0:
                print(f'Iteration {i}, Best Position: {self.global_best_position}, Fitness: {self.global_best_fitness:.6f}')

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Elapsed Time:', elapsed_time)
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
    pso = PSO(objective_function, bounds, num_particles, maxiter)
    pso.run_result()
    # pso.run_animation()