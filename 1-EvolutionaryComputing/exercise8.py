from multiprocessing.sharedctypes import Value
import operator
import math
import numpy
import copy
import random
import matplotlib.pyplot as plt
from deap import gp, base, tools, algorithms, creator

def safeDiv(x,y):
    try:
        return x / y;
    except ZeroDivisionError:
        return 1

def safeLog(x,y):
    try:
        return math.log(x,y)
    except (ValueError, ZeroDivisionError):
        return x

pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(safeLog, 2)
pset.addPrimitive(safeDiv, 2)
pset.addPrimitive(math.exp, 1)
pset.addPrimitive(math.sin, 1)

pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))

ypoints = [0.000, -0.1629, -0.2624, -0.3129, -0.3264, -0.3125, -0.2784, -0.2289, -0.1664, -0.909, 0.0, 0.1111, 0.2496, 0.4251, 0.6469, 0.9375, 1.3056, 1.7731, 2.3616, 3.0951, 4.0000]
xpoints = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# def fitness(x, y):
#     return abs(x-y),

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalSymbReg(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    abs_errors = (abs(func(xpoints[i]) - ypoints[i]) for i in range(len(xpoints)))

    sum = math.fsum(abs_errors)
    return (-sum),


toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("evaluate", evalSymbReg)
toolbox.register("select", tools.selTournament, tournsize=3)
# toolbox.register("evaluate", evalSymbReg, points=[x/10. for x in range(-10,10)])

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def plot(logbook, best_sizes):
    gen = logbook.select("gen")
    fit_max = logbook.chapters["fitness"].select("max")

    avg_sizes = logbook.chapters["size"].select("avg")
    min_sizes = logbook.chapters["size"].select("min")
    max_sizes = logbook.chapters["size"].select("max")
    

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_max, "b-", label="Best fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    ax2 = ax1.twinx()
    line2 = ax2.plot(gen, avg_sizes, "r-", label="Average Size")
    line3 = ax2.plot(gen, max_sizes, "darkred")
    line4 = ax2.plot(gen, min_sizes, "darkred")
    line5 = ax2.plot(gen, best_sizes, "lightcoral", label="Size best fitness")


    ax2.fill_between(gen, min_sizes, max_sizes, color='darkred', alpha=.5)

    ax2.set_ylabel("Size", color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    lns = line1 + line5
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")

    plt.show()

def main():
    random.seed(42)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    # stats_size_greatest = tools.Statistics(lambda ind: ind.size.values)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)
    # mstats.register("size_fit_max", lambda x: len(tools.selBest(x, 1)[0]))

    pop = toolbox.population(n=1000)
    # pop2 = copy.deepcopy(pop)
    hof = tools.HallOfFame(1)

    hardcoded_best_sizes = [2, 7, 7, 7, 7, 7, 5, 5, 5, 5, 7, 7, 7, 7, 12, 12, 13, 13, 13, 15, 15, 16, 16, 23, 22, 20, 20, 21, 20, 20, 20, 31, 30, 30, 30, 30, 30, 30, 42, 42, 62, 62, 62, 62, 37, 62, 62, 64, 63, 63, 67]
    ngen = 50


    
    # best_sizes = []
    # best_sizes.append(len(tools.selBest(pop, 1)[0]))
    # for _ in range(ngen):
    #     pop, ew = algorithms.eaSimple(pop, toolbox, 0.7, 0, 1, stats=mstats)
    #     bests = tools.selBest(pop, 1)

    #     best_sizes.append(len(bests[0]))

    # print("SIZE BEST SIZE")
    # print(best_sizes)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.7, 0, ngen, stats=mstats,
                                   halloffame=hof, verbose=True)
    
    plot(log, hardcoded_best_sizes)

    return pop, log, hof

if __name__ == "__main__":
    main()