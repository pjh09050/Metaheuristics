import random
import math

# TSP 문제를 위한 도시들의 좌표
cities = {
    0: (60, 200),
    1: (180, 200),
    2: (80, 180),
    3: (140, 180),
    4: (20, 160),
    5: (100, 160),
    6: (200, 160),
    7: (140, 140),
    8: (40, 120),
    9: (100, 120),
    10: (180, 100),
    11: (60, 80),
    12: (120, 80),
    13: (180, 60),
    14: (20, 40),
    15: (100, 40),
    16: (200, 40),
    17: (20, 20),
    18: (60, 20),
    19: (160, 20)
}

def calculate_distance(city1, city2):
    # 두 도시 간의 거리 계산
    x1, y1 = cities[city1]
    x2, y2 = cities[city2]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def evaluate_solution(solution):
    # 경로의 총 거리 계산
    total_distance = 0
    num_cities = len(solution)
    for i in range(num_cities):
        total_distance += calculate_distance(solution[i], solution[(i+1) % num_cities])
    return total_distance

def generate_initial_solution():
    # 무작위 초기 해 생성
    solution = list(cities.keys())
    random.shuffle(solution)
    return solution

def shake(solution, k_max):
    # 현재 해를 k_max 범위 내의 무작위한 변동을 통해 흔들어주는 함수
    k = random.randint(1, k_max)
    new_solution = solution[:]
    for i in range(k):
        index1 = random.randint(0, len(new_solution) - 1)
        index2 = random.randint(0, len(new_solution) - 1)
        new_solution[index1], new_solution[index2] = new_solution[index2], new_solution[index1]
    return new_solution

def two_opt(local_solution):
    # 2-opt 교환을 통한 국부 탐색
    best = local_solution
    improved = True
    while improved:
        improved = False
        for i in range(1, len(local_solution) - 1):
            for j in range(i + 1, len(local_solution)):
                if j - i == 1:
                    continue  # No point in swapping adjacent elements
                new_route = local_solution[:]
                new_route[i:j] = local_solution[j - 1:i - 1:-1]  # reverse segment
                if evaluate_solution(new_route) < evaluate_solution(best):
                    best = new_route
                    improved = True
        local_solution = best
    return best

def variable_neighborhood_search(max_iterations, k_max):
    # Variable Neighborhood Search 알고리즘
    best_solution = generate_initial_solution()
    best_fitness = evaluate_solution(best_solution)
    
    iteration = 0
    while iteration < max_iterations:
        k = 1
        while k <= k_max:
            new_solution = shake(best_solution, k)
            new_solution = two_opt(new_solution)
            new_fitness = evaluate_solution(new_solution)
            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
                k = 1  # 다시 첫 번째 이웃으로 돌아감
            else:
                k += 1  # 다음 이웃으로 이동
        
        iteration += 1
    
    return best_solution, best_fitness

# 실행 예시
best_solution, best_fitness = variable_neighborhood_search(max_iterations=100, k_max=5)
print("Best solution:", best_solution)
print("Best fitness:", best_fitness)
