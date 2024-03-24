import numpy as np
import random

# TSP 문제의 도시들을 무작위로 생성하는 함수
def generate_cities(num_cities, max_coord):
    return np.random.randint(max_coord, size=(num_cities, 2))

# 두 도시 간의 거리를 계산하는 함수 (유클리드 거리 사용)
def distance(city1, city2):
    return np.linalg.norm(city1 - city2)

# 초기해를 무작위로 생성하는 함수
def generate_initial_solution(num_cities):
    return list(range(num_cities))

# 현재 해를 k_max 범위 내의 무작위한 변동을 통해 흔들어주는 함수
def shake(solution, k_max):
    new_solution = solution[:]
    for _ in range(k_max):
        idx1, idx2 = random.sample(range(len(solution)), 2)
        new_solution[idx1], new_solution[idx2] = new_solution[idx2], new_solution[idx1]
    return new_solution

# 국부 탐색을 수행하는 함수 (2-opt 방법 사용)
def local_search(solution, cities):
    improved = True
    best_solution = solution[:]
    best_distance = calculate_total_distance(solution, cities)
    while improved:
        improved = False
        for i in range(1, len(solution) - 2):
            for j in range(i + 1, len(solution)):
                if j - i == 1:
                    continue  # 인접한 경로는 건너뜀
                new_solution = solution[:]
                new_solution[i:j] = solution[j - 1:i - 1:-1]  # 뒤집기
                new_distance = calculate_total_distance(new_solution, cities)
                if new_distance < best_distance:
                    best_distance = new_distance
                    best_solution = new_solution[:]
                    improved = True
        solution = best_solution[:]
    return best_solution

# 주어진 해의 품질을 계산하는 함수 (총 이동 거리)
def calculate_total_distance(solution, cities):
    total_distance = 0
    for i in range(len(solution)):
        total_distance += distance(cities[solution[i]], cities[solution[(i + 1) % len(solution)]])
    return total_distance

# Variable Neighborhood Search 알고리즘
def variable_neighborhood_search(cities, max_iterations, k_max):
    num_cities = len(cities)
    best_solution = generate_initial_solution(num_cities)
    best_distance = calculate_total_distance(best_solution, cities)
    
    iteration = 0
    while iteration < max_iterations:
        k = 1
        while k <= k_max:
            new_solution = shake(best_solution, k)
            new_solution = local_search(new_solution, cities)
            new_distance = calculate_total_distance(new_solution, cities)
            if new_distance < best_distance:
                best_solution = new_solution[:]
                best_distance = new_distance
                k = 1  # 다시 첫 번째 이웃으로 돌아감
            else:
                k += 1  # 다음 이웃으로 이동
        
        iteration += 1
    
    return best_solution, best_distance

# 실행 예시
np.random.seed(0)
num_cities = 10
max_coord = 100
cities = generate_cities(num_cities, max_coord)
best_solution, best_distance = variable_neighborhood_search(cities, max_iterations=100, k_max=5)
print("Best solution order:", best_solution)
print("Best total distance:", best_distance)