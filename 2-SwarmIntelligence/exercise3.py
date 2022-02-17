from multiprocessing.sharedctypes import Value
import operator
import math
from re import X
import numpy
import copy
import random
import matplotlib.pyplot as plt

iterations = 10

# Initial values
init_speed = 0

N = 5
no_particles = 10


def gen_dataset_1():
    vectors = []
    
    def classify(z1, z2):
        if (z1 >= 0.7 or z1 <= 0.3) and (z2 >= -0.2 - z1):
            return 1
        return 0

    for _ in range(400):
        z1 = random.uniform(-1, 1)
        z2 = random.uniform(-1, 1)
        
        rand_vec = (z1, z2)
        vectors.append((rand_vec, classify(z1, z2)))
    
    return vectors


def optimizePSO():
    return

def kMeans():
    particles = []

    data = gen_dataset_1()
    for i in range(no_particles):
        
    
    
    return
    
def calc_fitness_8():
    return 0

def run(dataset, N):
    particles = []

    data = gen_dataset_1(dataset)
    for i in range(no_particles):
        particle = random.sample(data, N)
        particles.append(particle)
    
    for i in range(iterations):
        for x in particles:
            for z in data:
                distance = None
                cluster = None
                for c in x:
                    new_distance = math.sqrt((c[0][0]-z[0][0])**2 + (c[0][1] - z[0][1])**2)
                    if distance is None or distance > new_distance:
                        distance = new_distance
                        cluster = c
                z[1] = cluster
            
            fitness()
                


        
        
        
        

def plot():
    # run1 =  run(w_1, a1_1, a2_1, r1_1, r2_1)
    # run2 = run(w_2, a1_2,a2_2,r1_2,r2_2)
   


    # for x in range(iterations):
    #     plt.plot(range(iterations), run1[x], label = f"Run1", color = 'green')
    #     plt.plot(range(iterations), run2[x], label = f"Run2", color = 'red') 
    # plt.plot(range(iterations+1), run1, label = f"Run1", color = 'green')
    # plt.plot(range(iterations+1), run2, label = f"Run2", color = 'red')   
    # plt.plot(range(iterations), avgMA, label="avgMA")
    # plt.legend()  
    # plt.show()



if __name__ == "__main__":
    plot()