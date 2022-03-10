import csv
import math
from cv2 import sumElems, transpose
from matplotlib import pyplot as plt
import numpy as np
import random
       

""""
Particle is a list of cluster centroids that is being optimized
"""
class Particle:
    
    def __init__(self, data, amount_classes):
        self.centroids = np.array(random.sample(list(data), amount_classes))
  
        self.velocity = np.zeros_like(self.centroids)
        self.local_best = self.centroids
        self.clusters = [[] for _ in self.centroids]

        self.w = 0.72
        self.c1 = 1.49
        self.c2 = 1.49
        self.r1 = random.uniform(0,1)
        self.r2 = random.uniform(0,1)
        # Indeed every particle has list of cluster and within this particle you update the location of the clusters based on their fit to the datapoints (this is the local best). 
        # Then you can compare the outcomes of the particles and find the global best. 
        # So you should indeed calculate for all datapoints their distance to the centroids in a particle, and do this for all particle. 

    def update_velocity(self, global_best):
        self.r1 = random.uniform(0,1)
        self.r2 = random.uniform(0,1)
        new_velocity = self.w * self.velocity + np.dot(self.c1*self.r1,(self.local_best - self.centroids)) + np.dot(self.c2*self.r2,(global_best - self.centroids))
        print("new velocity = ", new_velocity)
        self.velocity = new_velocity

    def update_centroids(self):
        self.centroids = self.centroids + self.velocity
    
    def update_local_best(self):
        result = []
        for i in range(len(self.centroids)):
            cluster = self.clusters[i]
            # print(len(cluster))
            result.append((np.dot(1/(len(cluster)), np.sum(np.array(cluster), axis =0))).tolist())
        self.local_best = result
        print("Local Best = ", self.local_best)
        

    def reset_clusters(self):
        self.clusters = [[] for _ in self.clusters]

    
    def calc_euclidean_distance(self, c, z):
        return math.sqrt(sum((np.array(c)-np.array(z))**2))

    def calc_fitness_8(self):
        sum = 0
        for c in range(len(self.centroids)):
            cluster = self.clusters[c]
            print("cluster = ", cluster)
            for z in cluster:
                sum += (self.calc_euclidean_distance(self.centroids[c], z) / len(cluster))
        print("sum = ", sum)
        return sum/len(self.centroids)

# ======= MAIN =======================================

iterations = 10
NParticles = 10
trials = 2

def gen_dataset_1():
    vectors = []
    for _ in range(400):
        z1 = random.uniform(-1, 1)
        z2 = random.uniform(-1, 1)
        
        vectors.append([z1, z2])
    return vectors  

def gen_dataset_2(path):
    vectors = []
    with open(path) as f:
        vals = [[num for num in line.split(",")] for line in f]
        for row in vals:
            print(row)
            vec = [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
            vectors.append(vec)  
    return vectors    

def plot(fitness):
    for p in range(NParticles):
        plt.plot(range(iterations), np.transpose(fitness)[p], label = f"P{p}")
    plt.legend()  
    plt.show()
    
def runPSO(dataset, N):
    # data = dataset
    particles = []

    # 1 initialize each particle to contain N_c randomly selected cluster centroids
    for _ in range(NParticles):
        particles.append(Particle(dataset, N))
    print("Particles = ", particles)
    globalBest = particles[0].centroids
    

    fitnesses = []
    # 2 for t=1 to t_max (max iterations) do
    for i in range(iterations):
        # (a) for each particle i do
        fitness_iteration = []
        for i in particles:
            # (b) for each datavector z_p
            print("i = ", i.centroids)
            for z in dataset:
                # i calculate the Euclidean distance d(z_p, m_ij) to all cluster centroids c_ij
                distance = None
                cluster = None
                for j in range(len(i.centroids)):
                    c = i.centroids[j]
                    new_distance = Particle.calc_euclidean_distance(i,c,z)
                    if distance is None or distance > new_distance:
                        distance = new_distance
                        cluster = j
                
                # ii assign z_p to cluster C_ij with minimal distance
                i.clusters[cluster].append(z)
            # iii calculate fitness using equation (8)
            fitness_iteration.append(Particle.calc_fitness_8(i))
            # (c) update local best
            Particle.update_local_best(i)
        # (c) update global best
        globalBest = particles[np.where(fitness_iteration == np.amin(fitness_iteration))[0][0]].centroids
        print("GlobalBest = ", globalBest)
        # (d) update cluster centroids using velocity and position update rules
        for p in particles:
            Particle.update_velocity(p, globalBest)
            Particle.update_centroids(p)
            Particle.reset_clusters(p)
        fitnesses.append(fitness_iteration)
        print("Fitnesses = ", fitnesses)
    plot(fitnesses)
        
# data = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
data = gen_dataset_2("iris.data")
runPSO(data, 2)
# m = data.sum(axis=0)
# Particle.update_centroids(a)
# print(a.centroids)

# e = a.calc_euclidean_distance(a.centroids[0],data[0])
# print(e)

# f = Particle.calc_fitness_8(a)
# print(f)

# print(2*3*(a.local_best + a.centroids))
# Particle.update_velocity(a, a.centroids)
# print(a.velocity)
 