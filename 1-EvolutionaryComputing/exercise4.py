import random
import matplotlib.pyplot as plt
import numpy as np


l = 100
p = 1 / l
iterations = 1500
runs = 10
log = False #Set whether you want to show print statements.

def bitsToInt(x):
    multiplier = l - 1
    sum = 0
    for i in x:
        sum += i * 2**multiplier
        multiplier -= 1
    return sum    

def fitness(x, goal):
    return abs(bitsToInt(goal) - bitsToInt(x))

def generateX():
    return [random.randint(0,1) for _ in range(l)]

def invert(x):
    x_m = []
    for i in x:
        if random.random() < p:
            if i == 0:
                x_m.append(1)
            elif i == 1:
                x_m.append(0)
        else:
            x_m.append(i)
    return x_m

def run_ab():
    results = []
    goal = generateX()
    if log: print(goal)
    x = generateX()
    for i in range(iterations):
        x_m = invert(x)
        if fitness(x_m, goal) < fitness(x, goal):
            x = x_m
        if log: print(f"{i} : {fitness(x, goal)}")
        results.append(fitness(x, goal))
    
    if log: 
        print(f"goal : {goal}")  
        print(f"reached x Bit : {x}")
        print(f"goalBit : {bitsToInt(goal)}")
        print(f"reached x : {bitsToInt(x)}")
        print(f"reached fitness : {fitness(x, goal)}")

    return results

def run_c():
    results = []
    goal = generateX()
    if log: print(goal)
    x = generateX()
    for i in range(iterations):
        x_m = invert(x)
        x = x_m
        if log: print(f"{i} : {fitness(x, goal)}")
        results.append(fitness(x, goal))

    if log: 
        print(f"goal : {goal}")  
        print(f"reached x Bit : {x}")
        print(f"goalBit : {bitsToInt(goal)}")
        print(f"reached x : {bitsToInt(x)}")
        print(f"reached fitness : {fitness(x, goal)}")
    return results

# Plot results of assignment 4a
def plot_chart_a():
    plt.plot(range(iterations), run_ab(), label = f"run")  
    plt.legend()
    plt.show()

# Plot results of assignment 4b
def plot_chart_b():
    for i in range(runs):
        plt.plot(range(iterations), run_ab(), label = f"run {i}")    
    plt.legend()
    plt.show()

# Plot results of assignment 4c
def plot_chart_c():
    for i in range(runs):
        plt.plot(range(iterations), run_c(), label = f"run {i}")    
    plt.legend()
    plt.show()


plot_chart_a()
plot_chart_b()
plot_chart_c()