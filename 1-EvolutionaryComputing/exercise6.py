import random
import matplotlib.pyplot as plt
import numpy
import math

iterations = 1500

# Calculates the total distance of a route
def calculate_total_distance(route, cities):
    distance = 0
    for i in range(1, len(route)-1):
        a = cities[route[i-1]]
        b = cities[route[i]]
        distance += math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    return distance

# Returnes the fitness of an individual based on the provided calculated total distance
def fitness(distance):
    return 1 / distance

# Fills in the missing city indexes after the crossover operation
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

# The crossover operation
def crossover(parent1, parent2):
    cut1 = random.randint(0, len(parent1)-1)
    cut2 = random.randint(cut1+1, len(parent1))
   
    ofs1 = [None] * len(parent1)
    ofs2 = [None] * len(parent2)

    ofs1[cut1:cut2] = parent1[cut1:cut2]
    ofs2[cut1:cut2] = parent2[cut1:cut2]
    
    ofs1 = crosscut(parent2, ofs1, cut1, cut2)
    ofs2 = crosscut(parent1, ofs2, cut1, cut2)

    return ofs1, ofs2

# The Reverse Sequence Mutation operation
def rsm(route):
    cuts = random.sample(range(0, len(route)), 2)

    value2 = route[cuts[1]]
    route[cuts[1]] = route[cuts[0]]
    route[cuts[0]] = value2
    return route

# The key used to sort the population lists, based on their fitness
def key(cities):
    return lambda route: fitness(calculate_total_distance(route, cities))

def runEA(cities):
    population = []
    for i in range(4):
        route = random.sample(range(len(cities)), len(cities))
        population.append(route)
    
    best_fitnesses = []
    avg_fitnesses = []

    for i in range(iterations):
        print(population)
        print(f"{i} : {key(cities)(population[0])}")
        list.sort(population, key=key(cities=cities), reverse=True)
        (s1,s2) = crossover(population[0], population[1])
        best_fitnesses.append(key(cities)(population[0]))
        avg_fitnesses.append(calc_avg_fitness(population, cities))
        population[2] = rsm(s1)
        population[3] = rsm(s2)

    list.sort(population, key=key(cities=cities), reverse=True)
    print(f"Best route: {population[0]}")
    print(f"Best fitness: {key(cities)(population[0])}")
    return best_fitnesses, avg_fitnesses

# Runs the Memetic Algorithm
def runMA(cities):
    population = []
    for i in range(4):
        route = random.sample(range(len(cities)), len(cities))
        population.append(ma(route,cities))
    
    best_fitnesses = []
    avg_fitnesses = []

    for i in range(iterations):
        print(population)
        print(f"{i} : {key(cities)(population[0])}")
        list.sort(population, key=key(cities=cities), reverse=True)
        (s1,s2) = crossover(population[0], population[1])
        best_fitnesses.append(key(cities)(population[0]))
        avg_fitnesses.append(calc_avg_fitness(population, cities))
        population[2] = ma(rsm(s1), cities)
        population[3] = ma(rsm(s2),cities)
        
    list.sort(population, key=key(cities=cities), reverse=True)
    print(f"Best route: {population[0]}")
    print(f"Best fitness: {key(cities)(population[0])}")
    return best_fitnesses, avg_fitnesses

# Computes the average fitness of a population
def calc_avg_fitness(population,cities):
    fitnesses = [key(cities)(x) for x in population]
    return sum(fitnesses) / len(fitnesses)

# The Local Search operation based on the 2-opt algorithm, which is part of the Memetic Algorithm
def ma(route,cities):
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
            
            
# The operation that swaps 2 cities according to the 2-opt algorithm
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

# Plots the graphs for exercise 6 c.
def plot_chart(path, split):
    cities = readTSPData(path, split)
    runs = 10

    bestMA = []
    avgMA = []
    for i in range (runs):
        best, avg = runMA(cities)
        bestMA.append(best)
        avgMA.append(avg)

    bestEA = []
    avgEA = []
    for i in range (runs):
        best, avg = runEA(cities)
        bestEA.append(best)
        avgEA.append(avg)

    # avgMA = [0] * iterations # Average best fitness per iteration over the 10 runs
    # avgEA = [0] * iterations 

    # for i in range(iterations):
    #     for r in range(runs):
    #         avgMA[i] += bestMA[r][i]
    #         avgEA[i] += runsEA[r][i]
    #     avgMA[i] = avgMA[i]/runs
    #     avgEA[i] = avgEA[i]/runs

    # avgMA = [sum(x) / len(x) for x in runsMA]
    # avgEA = [sum(x) / len(x) for x in runsEA]

    ax = plt.gca()

    for x in range(runs):
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(range(iterations), bestMA[x], label = f"bestMA {x}", color = color)
        plt.plot(range(iterations), avgMA[x], label = f"avgMA {x}", color = color, linestyle="dashed")   
    # plt.plot(range(iterations), avgMA, label="avgMA")
    plt.legend(bbox_to_anchor=(1.04, 1.0), loc='upper left')
    plt.tight_layout()   
    plt.show()

    ax = plt.gca()
    for x in range(runs):
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(range(iterations), bestEA[x], label = f"bestEA {x}", color = color)
        plt.plot(range(iterations), avgEA[x], label = f"avgEA {x}", color = color, linestyle="dashed")   
    # plt.plot(range(iterations), avgEA[x], label="avgEA")
    plt.legend(bbox_to_anchor=(1.04, 1.0), loc='upper left')
    plt.tight_layout()
    plt.show()
    
# Reads the TSP Data from text files, given the file path and the split used in the document
def readTSPData(path, split):

    with open(path) as f:
        return [[float(num) for num in line.split(split)] for line in f]
        print(cities)


#The main function calls
plot_chart("bier127.txt", "  ")
plot_chart("file-tsp.txt", "   ")
