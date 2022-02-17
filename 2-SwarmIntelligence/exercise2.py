from multiprocessing.sharedctypes import Value
import operator
import math
import numpy
import copy
import random
import matplotlib.pyplot as plt

iterations = 10

# Initial values
initial_v = 10.0
initial_x = 20.0

# Settings 1
w_1 = 0.5
a1_1 = 1.5
a2_1 = 1.5
r1_1 = 0.5
r2_1 = 0.5

# Settings 2
w_2 = 0.7
a1_2 = 1.5
a2_2 = 1.5
r1_2 = 1.0
r2_2 = 1.0



def fitness(x):
    return x**2

def velocityCalc(w, v, a1, a2, r1, r2, x_local, x, x_global):
    return w * v + a1 * r1 *(x_local - x) + a2*r2*(x_global - x)

def run(w, a1, a2, r1, r2):
    result = [] 
    particle = initial_x
    result.append(fitness(particle))
    velocity = initial_v

    globalB = initial_x
    localB = initial_x

    for i in range(iterations):
        velocity = velocityCalc(w, velocity, a1, a2, r1, r2, localB, particle, globalB)
        particle = particle + velocity
        result.append(fitness(particle))
        if fitness(particle) < fitness(localB):
            localB = particle
        if fitness(particle) < fitness(globalB):
            globalB = particle

    return result
        

def plot():
    run1 =  run(w_1, a1_1, a2_1, r1_1, r2_1)
    run2 = run(w_2, a1_2,a2_2,r1_2,r2_2)


    # for x in range(iterations):
    #     plt.plot(range(iterations), run1[x], label = f"Run1", color = 'green')
    #     plt.plot(range(iterations), run2[x], label = f"Run2", color = 'red') 
    plt.plot(range(iterations+1), run1, label = f"Run1", color = 'green')
    plt.plot(range(iterations+1), run2, label = f"Run2", color = 'red')   
    # plt.plot(range(iterations), avgMA, label="avgMA")
    plt.legend()  
    plt.show()



if __name__ == "__main__":
    plot()