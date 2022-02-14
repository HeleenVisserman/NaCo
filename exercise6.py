import random
import matplotlib.pyplot as plt
import numpy as np
import math

# distances = [[]]
# cities = [[0.2554,18.2366], [0.4339,15.2476], [0.7377,8.3137], [1.1354,16.5638]]
# route= [0, 8, 4]

# route2 = [(x1,y1),(x2.y2)]

iterations = 1500
# cities = [None]

def calculate_total_distance(route, cities):
    distance = 0
    for i in range(1, len(route)-1):
        a = cities[route[i-1]]
        b = cities[route[i]]
        distance += math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
        # distance += np.linalg.norm(a-b)   #route[i-1] + route[i]
    return distance

def fitness(distance):
    return 1 / distance

def crosscut(parent, ofs, cut1, cut2):
    ofsRes = ofs
    pi = 0
    for i in range(len(parent)):
        if(i >= cut1 and i < cut2):
            continue
        else:
            for j in range(pi, len(parent)):
                if parent[j] not in ofsRes:
                    ofsRes[i] = parent[j]
                    pi = j
                    break
    return ofsRes

def crossover(parent1, parent2):
    cut1 = random.randint(0, len(parent1)-1)
    cut2 = random.randint(cut1+1, len(parent1))
   
    # print(cut1)
    # print(cut2)

    ofs1 = [None] * len(parent1)
    ofs2 = [None] * len(parent2)

    ofs1[cut1:cut2] = parent1[cut1:cut2]
    ofs2[cut1:cut2] = parent2[cut1:cut2]
    # print(ofs1)
    
    ofs1 = crosscut(parent2, ofs1, cut1, cut2)
    ofs2 = crosscut(parent1, ofs2, cut1, cut2)

    # print(ofs1)
    return ofs1, ofs2

def rsm(route):
    cuts = random.sample(range(0, len(route)), 2)

    value2 = route[cuts[1]]
    route[cuts[1]] = route[cuts[0]]
    route[cuts[0]] = value2
    return route

def key(cities):
    return lambda route: fitness(calculate_total_distance(route, cities))

def runEA(cities):
    population = []
    for i in range(4):
        route = random.sample(range(len(cities)), len(cities))
        # fitness = fitness(calculate_total_distance(route))
        population.append(route)
    
    best_fitnesses = []

    for i in range(iterations):
        print(population)
        print(f"{i} : {key(cities)(population[0])}")
        list.sort(population, key=key(cities=cities), reverse=True)
        (s1,s2) = crossover(population[0], population[1])
        best_fitnesses.append(key(cities)(population[0]))
        population[2] = rsm(s1)
        population[3] = rsm(s2)

    list.sort(population, key=key(cities=cities), reverse=True)
    print(f"Best route: {population[0]}")
    print(f"Best fitness: {key(cities)(population[0])}")
    return best_fitnesses


def runMA(cities):
    population = []
    for i in range(4):
        route = random.sample(range(len(cities)), len(cities))
        # fitness = fitness(calculate_total_distance(route))
        population.append(ma(route,cities))
    
    best_fitnesses = []

    for i in range(iterations):
        print(population)
        print(f"{i} : {key(cities)(population[0])}")
        list.sort(population, key=key(cities=cities), reverse=True)
        (s1,s2) = crossover(population[0], population[1])
        best_fitnesses.append(key(cities)(population[0]))
        population[2] = ma(rsm(s1), cities)
        population[3] = ma(rsm(s2),cities)
        
    list.sort(population, key=key(cities=cities), reverse=True)
    print(f"Best route: {population[0]}")
    print(f"Best fitness: {key(cities)(population[0])}")
    return best_fitnesses

def calc_avg_fitness(population,cities):
    fitnesses = [key(cities)(x) for x in population]
    return sum(fitnesses) / len(fitnesses)


def ma(route,cities):
    # route = random.sample(range(len(cities)), len(cities))
    existing_route = route
    best_distance = key(cities)(existing_route)
    for i in range(len(cities)-1):
        for j in range(i+1, len(cities)):
            new_route = optSwap(existing_route, i, j)
            new_distance = calculate_total_distance(new_route, cities)
            if (new_distance < best_distance):
                existing_route = new_route
                best_distance = new_distance
                break
    return existing_route
            
            

def optSwap(route, i, j):
    newRoute = [None] * len(route)
    for k in range(0,i):
        newRoute[k] = route[k]
    l = j
    for k in range(i,j+1):
        newRoute[k] = route[l]
        l -= 1
    for k in range(j+1,len(route)):
        newRoute[k] = route[k]
    return newRoute

def plot_chart(path, split):
    cities = readTSPData(path, split)
    runs = 10
    runsMA = [runMA(cities) for _ in range(runs)]
    runsEA = [runEA(cities) for _ in range(runs)]

    avgMA = [0] * iterations # Average best fitness per iteration over the 10 runs
    avgEA = [0] * iterations 

    for i in range(iterations):
        for r in range(runs):
            avgMA[i] += runsMA[r][i]
            avgEA[i] += runsEA[r][i]
        avgMA[i] = avgMA[i]/runs
        avgEA[i] = avgEA[i]/runs




    # avgMA = [sum(x) / len(x) for x in runsMA]
    # avgEA = [sum(x) / len(x) for x in runsEA]

    for x in range(runs):
        plt.plot(range(iterations), runsMA[x], label = f"runMA {x}")   
    plt.plot(range(iterations), avgMA, label="avgMA")
    plt.legend()
    plt.show()

    for x in range(runs):
        plt.plot(range(iterations), runsEA[x], label = f"runEA {x}")
    plt.plot(range(iterations), avgEA, label="avgEA")
    plt.legend()
    plt.show()
    

def readTSPData(path, split):

    with open(path) as f:
        return [[float(num) for num in line.split(split)] for line in f]
        print(cities)

    # with open(path) as f:
    #     lines = f.readlines()
    # coordinates= []
    # count = 0
    # for line in lines:
    #     coordinates.append(line.split("\s"))
    #     count += 1
    #     print(f'line {count}: {line}') 

plot_chart("bier127.txt", "  ")
plot_chart("file-tsp.txt", "   ")
# runMA()


# procedure 2optSwap(route, i, j) {
#     1. take route[0] to route[i-1] and add them in order to new_route
#     2. take route[i] to route[j] and add them in reverse order to new_route
#     3. take route[j+1] to end and add them in order to new_route
#     return new_route;
# }


    
# repeat until no improvement is made {
#     best_distance = calculateTotalDistance(existing_route)
#     start_again:
#     for (i = 0; i <= number of nodes eligible to be swapped - 1; i++) {
#         for (k = i + 1; k <= number of nodes eligible to be swapped; k++) {
#             new_route = 2optSwap(existing_route, i, k)
#             new_distance = calculateTotalDistance(new_route)
#             if (new_distance < best_distance) {
#                 existing_route = new_route
#                 best_distance = new_distance
#                 goto start_again
#             }
#         }
#     }
# }