import numpy as np

class GWO:
    def __init__(self, obj_func, bounds, dim, num_particles, max_iter):
        self.obj_func = obj_func
        self.bounds = bounds
        self.dim = dim
        self.num_particles = num_particles
        self.max_iter = max_iter

    def optimize(self):
        alpha_pos = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
        beta_pos = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
        delta_pos = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
        
        # Initialize the positions of search agents
        positions = np.zeros((self.num_particles, self.dim))
        for i in range(self.num_particles):
            for d in range(self.dim):
                positions[i, d] = np.random.uniform(self.bounds[d][0], self.bounds[d][1])

        for t in range(self.max_iter):
            # Update the positions of alpha, beta, and delta wolves
            for i in range(self.num_particles):
                fitness = self.obj_func(positions[i, :])
                if fitness < self.obj_func(alpha_pos):
                    alpha_pos = positions[i, :]
                if (fitness > self.obj_func(alpha_pos)) and (fitness < self.obj_func(beta_pos)):
                    beta_pos = positions[i, :]
                if (fitness > self.obj_func(alpha_pos)) and (fitness > self.obj_func(beta_pos)) and (fitness < self.obj_func(delta_pos)):
                    delta_pos = positions[i, :]

            # Update the position of search agents
            a = 2 - 2 * (t / self.max_iter) # parameter to control the search step size
            for i in range(self.num_particles):
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                A1 = 2 * a * r1 - a 
                C1 = 2 * r2 
                D_alpha = abs(C1 * alpha_pos - positions[i, :]) 
                X1 = alpha_pos - A1 * D_alpha 

                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                A2 = 2 * a * r1 - a 
                C2 = 2 * r2 
                D_beta = abs(C2 * beta_pos - positions[i, :]) 
                X2 = beta_pos - A2 * D_beta 

                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                A3 = 2 * a * r1 - a 
                C3 = 2 * r2 
                D_delta = abs(C3 * delta_pos - positions[i, :]) 
                X3 = delta_pos - A3 * D_delta 

                positions[i, :] = (X1 + X2 + X3) / 3 

            best_pos = alpha_pos
            best_score = self.obj_func(alpha_pos)

            if t % 10 == 0:
                print(f'Iteration {t}, Best Position: {best_pos}, Fitness: {best_score:.6f}')
                
        print('Best position:', best_pos)
        print('Best fitness:', best_score)

        return best_pos, best_score

def objective_function(x):
    x = np.array(x)
    y = np.array(x)
    z = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0]**2 + y[1]**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * y[1]))) + np.exp(1) + 20
    return z

if __name__ == "__main__":
    bounds = [(-4,4), (-4,4)]
    dim = len(bounds)
    num_particles = 10
    max_iter = 100
    gwo = GWO(objective_function, bounds, dim, num_particles, max_iter)
    gwo.optimize()
