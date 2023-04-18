# ------ GA Programming -----
# 00000 00000부터 11111 11111까지 가장 큰 이진 정수를 GA로 찾기
# 탐색 중에 해집단의 해들이 일정 비율 동일하게 수렴하면 최적 해로 수렴했다고 판단하고 탐색을 종료하도록 설계
# ---------------------------

# ----- 제약사항 ------
# pandas 모듈 사용 금지
# random 모듈만 사용, 필요시 numpy 사용 가능
# [chromosome, fitness]로 구성된 list 타입의 해 사용: ["1010", 10]
# population 형태는 다음과 같이 list 타입으로 규정: [["1010", 10], ["0001", 1], ["0011", 3]]
# --------------------

import random
import numpy
import matplotlib.pyplot as plt

# ----- 수정 가능한 파라미터 -----

params = {
    'MUT': 0.5,  # 변이확률
    'END' : 0.9,  # 설정한 비율만큼 chromosome이 수렴하면 탐색을 멈추게 하는 파라미터
    'POP_SIZE' : 100,  # population size 10 ~ 100
    'RANGE' : 10, # chromosome의 표현 범위, 만약 10이라면 00000 00000 ~ 11111 11111까지임
    'NUM_OFFSPRING' : 5, # 한 세대에 발생하는 자식 chromosome의 수
    'CHANGE' : 2 # 다음 세대로 가는 자식 교체 수
    # (원하는 파라미터는 여기에 삽입할 것)
    }
# ------------------------------

class GA():
    def __init__(self, parameters):
        self.params = {}
        for key, value in parameters.items():
            self.params[key] = value

    def get_fitness(self, chromosome):
        fitness = 0
        # todo: 이진수 -> 십진수로 변환하여 fitness 구하기
        fitness = int(chromosome,2)
        return fitness

    def print_average_fitness(self, population):
        # todo: population의 평균 fitness를 출력
        population_average_fitness = 0
        for i in range(len(population)):
            population_average_fitness += population[i][1]
        population_average_fitness = population_average_fitness / len(population)
        print("population 평균 fitness: {}".format(population_average_fitness))
        return population_average_fitness # 그래프 그리기 위한 return 

    def sort_population(self, population):
        # todo: fitness를 기준으로 population을 내림차순 정렬하고 반환
        population.sort(key=lambda x:-x[1])
        return population

    def selection_operater(self, population):
        # todo: 본인이 원하는 선택연산 구현(룰렛휠, 토너먼트, 순위 등), 선택압을 고려할 것, 한 쌍의 부모 chromosome 반환
        mom_ch = 0
        dad_ch = 0
        #--------------------------
        # 룰렛휠
        # fitness_percent = []
        # fitness_sum = 0
        # for i in range(len(population)):
        #     fitness_sum += population[i][1]
        # for i in range(len(population)):
        #     fitness_percent.append(population[i][1] / fitness_sum)
        # mom_ch, dad_ch = list(population[int(random.uniform(0, len(fitness_percent)))][0] for _ in range(2))
        #--------------------------
        # 토너먼트 (mom_ch : 좋은 것, dad_ch : 나쁜 것)
        t = 0.7
        for i in range(2):
            sample = random.sample(population, 2)
            sample = self.sort_population(sample)
            rand = random.uniform(0,1)
            if i == 0:
                if rand < t:
                    mom_ch = sample[0][0]
                else:
                    mom_ch = sample[1][0]
            if i == 1:
                if rand < t:
                    dad_ch = sample[0][0]
                else:
                    dad_ch = sample[1][0]
        #----------------------------
        # 순위
        # mom_ch = population[0][0]
        # dad_ch = population[-1][0]
        return mom_ch, dad_ch

    def crossover_operater(self, mom_cho, dad_cho):
        # todo: 본인이 원하는 교차연산 구현(point, pmx 등), 자식해 반환
        offspring_cho = 0
        mom_ch = list(mom_cho)
        dad_ch = list(dad_cho)
        k = random.randint(0, len(mom_ch))
        for i in range(k, len(mom_ch)):
            mom_ch[i], dad_ch[i] = dad_ch[i], mom_ch[i]
        mom_ch = ''.join(mom_ch)
        dad_ch = ''.join(dad_ch)
        if self.get_fitness(mom_ch) >= self.get_fitness(dad_ch):
            offspring_cho = mom_ch
        else:
            offspring_cho = dad_ch
        return offspring_cho

    def mutation_operater(self, chromosome):        
        # todo: 변이가 결정되었다면 chromosome 안에서 랜덤하게 지정된 하나의 gene를 반대의 값(0->1, 1->0)으로 변이
        k = random.randint(0, len(chromosome)-1) # 변이할 위치 랜덤으로 설정
        chromosome = list(chromosome) # 리스트로 변경한 후 변이 실행
        if chromosome[k] == '0':
            chromosome[k] = '1'
        else:
            chromosome[k] = '0'
        chromosome = ''.join(chromosome)
        result_chromosome = chromosome
        return result_chromosome

    def replacement_operator(self, population, offsprings):
        # todo: 생성된 자식해들(offsprings)을 이용하여 기존 해집단(population)의 해를 대치하여 새로운 해집단을 return
        result_population = []
        population = self.sort_population(population)
        # 자식해 집단 중 뽑고 싶은 자식 수를 파라미터로 받아 가장 안좋은 해 대체
        offsprings = random.sample(offsprings, self.params["CHANGE"])
        for i in range(len(offsprings)):
            population[-(i+1)] = offsprings[i]
        result_population = self.sort_population(population)
        return result_population

    # 해 탐색(GA) 함수
    def search(self):
        generation = 0  # 현재 세대 수
        population = [] # 해집단
        offsprings = [] # 자식해집단
        average = []

        # 1. 초기화: 랜덤하게 해를 초기화
        for i in range(self.params["POP_SIZE"]):
            # todo: random 모듈을 사용하여 랜덤한 해 생성, self.params["range"]를 사용할 것
            # todo: fitness를 구하는 함수인 self.get_fitness()를 만들어서 fitness를 구할 것
            # todo: 정렬함수인 self.sort_population()을 사용하여 population을 정렬할 것
            chromosome = format(random.randint(0, 2 ** self.params["RANGE"] - 1), 'b').zfill(self.params["RANGE"])
            fitness = self.get_fitness(chromosome)
            population.append([chromosome, fitness])
            population = self.sort_population(population)
        print("initialzed population : \n", population, "\n\n")

        while 1:
            offsprings = []
            for i in range(self.params["NUM_OFFSPRING"]):
                #offsprings = []                 
                # 2. 선택 연산
                mom_ch, dad_ch = self.selection_operater(population)

                # 3. 교차 연산
                offspring = self.crossover_operater(mom_ch, dad_ch)

                # 4. 변이 연산
                # todo: 변이 연산여부를 결정, self.params["MUT"]에 따라 변이가 결정되지 않으면 변이연산 수행하지 않음
                if random.uniform(0,1) >= self.params["MUT"]:
                    offspring = self.mutation_operater(offspring)
                offsprings.append([offspring,self.get_fitness(offspring)])

            # 5. 대치 연산
            population = self.replacement_operator(population, offsprings)
            generation += 1

            # self.print_average_fitness(population) # population의 평균 fitness를 출력함으로써 수렴하는 모습을 보기 위한 기능
            average.append(self.print_average_fitness(population)) # population의 평균 fitness 그래프를 그리기 위한 average에 추가

            # 6. 알고리즘 종료 조건 판단
            # todo population이 전체 중 self.params["END"]의 비율만큼 동일한 해를 갖는다면 수렴했다고 판단하고 탐색 종료
            if population.count(population[0]) >= len(population) * self.params["END"]: # END비율만큼 수렴하면 정지
                break
            
        # 최종적으로 얼마나 소요되었는지의 세대수, 수렴된 chromosome과 fitness를 출력
        print("탐색이 완료되었습니다. \t 최종 세대수: {},\t 최종 해: {},\t 최종 적합도: {}".format(generation, population[0][0], population[0][1]))
        print('최종 population :', population)
        # population의 평균 fitness 그래프
        plt.plot(average)
        plt.ylim(0, 2 ** self.params["RANGE"] * 1.1)
        plt.show()

if __name__ == "__main__":
    ga = GA(params)
    ga.search()