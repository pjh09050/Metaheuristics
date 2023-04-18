import numpy as np
from scipy.stats import levy

class CuckooSearch:
    def __init__(self, fitness_func, dim, num_nests, num_iterations, p):
        self.fitness_func = fitness_func
        self.dim = dim
        self.num_nests = num_nests
        self.num_iterations = num_iterations
        self.p = p

    def levy_flight(self, beta):
        u = np.random.normal(0, 1)
        v = np.random.normal(0, 1)
        step = u / np.power(np.abs(v), 1/beta)
        return step

    def generate_nests(self):
        self.nests = np.random.rand(self.num_nests, self.dim)

    def evaluate_fitness(self):
        self.fitness = np.zeros(self.num_nests)
        for i in range(self.num_nests):
            self.fitness[i] = self.fitness_func(self.nests[i])

    def get_best_nest(self):
        idx = np.argmin(self.fitness)
        self.best_nest = self.nests[idx]
        self.best_fitness = self.fitness[idx]

    def empty_nests(self):
        # 일부 둥지를 levy_flight로 생성된 새 둥지로 교체
        num_replaced = int(self.p * self.num_nests)
        new_nests = np.zeros((num_replaced, self.dim))
        for i in range(num_replaced):
            j = np.random.randint(self.num_nests)
            step = 0.05  # Step size of Levy flight(Levy flight의 단계 크기 : 새 둥지가 원래 둥지에서 얼마나 떨어져 있는지)
            beta = 1.5  # Parameter for Levy flight(Levy flight의 모양을 결정하는 매개변수)
            new_nest = self.nests[j] + step * self.levy_flight(beta) * (self.nests[np.random.randint(self.num_nests)] - self.nests[np.random.randint(self.num_nests)])
            # Apply boundary constraints
            new_nest = np.clip(new_nest, 0, 1) # np.clip(arrange, min, max)
            new_nests[i,:] = new_nest
        return new_nests

    def optimize(self):
        # 둥지를 생성함
        self.generate_nests()
        # fitness 
        self.evaluate_fitness()
        # 가장 좋은 해 찾기
        self.get_best_nest()

        # 반복
        for i in range(self.num_iterations):
            new_nests = self.empty_nests()
            # 새로운 해의 fitness 평가
            new_fitness = np.full(self.num_nests, np.inf)
            for j in range(len(new_nests)):
                new_fitness[j] = self.fitness_func(new_nests[j])
            # 더 나은 해 대체
            idx = np.where(new_fitness < self.fitness)
            self.nests[idx] = new_nests[idx]
            self.fitness[idx] = new_fitness[idx]
            # 최고의 해 찾기
            self.get_best_nest()

            if i % 10 == 0:
                print("Iteration {}: Best nest = {}, Best fitness = {}".format(i, self.best_nest, self.best_fitness))

        return self.best_nest, self.best_fitness

def objective_function(x):
    x = np.array(x)
    y = np.array(x)
    z = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0]**2 + y[1]**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * y[1]))) + np.exp(1) + 20
    return z

if __name__ == "__main__":
    dim = 2  # 문제의 차원
    num_nests = 30 
    num_iterations = 100  
    p = 0.2  # 둥지가 새 둥지로 대체될 확률
    cs = CuckooSearch(objective_function, dim, num_nests, num_iterations, p)
    cs.optimize()