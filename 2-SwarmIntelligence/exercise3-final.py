import numpy as np
import random
import math
import csv
from matplotlib import pyplot as plt

""""
K-Means clustering algorithm.
"""
class KMeans:
    """
    1. Randomly initialize the Nc clutser centroid vectors
    2. Repeat
       a) For each data vector, assign the vector to the class
       with the closest centroid vector.
       b) Recalculate the cluster centroid vectors until a
       stopping criterion is satisfied (max iterations).
    """
    def __init__(
        self,
        k: int,
        max_iterations: int = 100):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = [[] for _ in range(k)]
        self.clusters = [[] for _ in range(k)]

        self.best_centroids = None
        self.best_clusters = None
        self.best_total_distance = None

        self.SSE = None

    # def plot(self, fitness, runs):
    #     plt.scatter(range(runs), fitness, marker='o')
    #     plt.legend()  
    #     plt.show()

    def reset(self):
        self.centroids = [[] for _ in range(self.k)]
        self.clusters = [[] for _ in range(self.k)]

        self.best_centroids = None
        self.best_clusters = None
        self.best_total_distance = None

        self.SSE = None


    def fit(self, data: np.array):
        self.reset()
        self._init_centroids(data)
        fitnesses = []
        for _ in range(self.max_iterations):
            for v in data:
                self._calculate_distances(v)
            total_distance = self._calculate_sse()
            if self.best_total_distance is None or self.best_total_distance > total_distance:
                self.best_total_distance = total_distance
                self.best_clusters = self.clusters
                self.best_centroids = self.centroids
            
            fitnesses.append(self.best_total_distance)

            self._calculate_centroids()
            self.clusters = [[] for _ in range(self.k)]
        return fitnesses

    def _calculate_centroids(self):
        for c in range(self.k):
            centroid = np.mean(self.clusters[c], axis=0)
            self.centroids[c] = centroid

    def _init_centroids(self, data: np.array):
        """Initialize centroids."""
        self.centroids = np.array(random.sample(list(data), self.k))
        
    def _calculate_distances(self, data: np.array):
        """Calculate the distance between data and centroids."""
        distance = None
        cluster = None
        for c in range(self.k):
            new_distance = self.calc_euclidean_distance(data, self.centroids[c])
            if distance is None or distance > new_distance:
                distance = new_distance
                cluster = c
        self.clusters[cluster].append(data)

    def calc_euclidean_distance(self, c, z):
        return math.sqrt(sum((np.array(c)-np.array(z))**2))
    
    def _calculate_sse(self):
        """
        Calculate the sum of squared errors (SSE) so we can find the best clustering.
        Goal is to minimize the SSE.
        """
        total_distance = 0
        
        for c in range(self.k):
            total_distance_t = 0
            for d in self.clusters[c]:
                total_distance += self.calc_euclidean_distance(d,self.centroids[c])
            total_distance = total_distance/len(self.clusters[c])
            total_distance_t += total_distance

        return total_distance_t / self.k

    
    
def update_centroids_kmeans(self):
    for i in range(len(self.centroids)):
        cluster = self.clusters[i].cluster
        self.centroids[i] = 1/(len(cluster)) * cluster.sum(axis=0)

def gen_dataset_1():
    vectors = []
    for _ in range(400):
        z1 = random.uniform(-1, 1)
        z2 = random.uniform(-1, 1)
        
        vectors.append([z1, z2])
    return vectors 


""""
Particle is a list of cluster centroids that is being optimized
"""
# global PSO variables
iterations = 100
NParticles = 10

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

    def update_velocity(self, global_best):
        self.r1 = random.uniform(0,1)
        self.r2 = random.uniform(0,1)
   
        new_velocity = self.w * self.velocity + np.dot(self.c1*self.r1,(self.local_best - self.centroids)) + np.dot(self.c2*self.r2,(global_best - self.centroids))
        self.velocity = new_velocity

    def update_centroids(self):
        self.centroids = self.centroids + self.velocity
    
    def update_local_best(self):
        result = []
        for i in range(len(self.centroids)):
            cluster = self.clusters[i]
            if len(cluster) != 0:
                result.append((np.dot(1/(len(cluster)), np.sum(np.array(cluster), axis =0))).tolist())
            else:
                result.append(self.centroids[i])
        self.local_best = result
        

    def reset_clusters(self):
        self.clusters = [[] for _ in self.clusters]

    
    def calc_euclidean_distance(self, c, z):
        return math.sqrt(sum((np.array(c)-np.array(z))**2))

    def calc_fitness_8(self):
        sum = 0
        for c in range(len(self.centroids)):
            cluster = self.clusters[c]
            for z in cluster:
                sum += (self.calc_euclidean_distance(self.centroids[c], z) / len(cluster))
        return sum/len(self.centroids)

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
        del vals[len(vals)-1]
        for row in vals:
            vec = [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
            vectors.append(vec)  
    return vectors    
            

def plot(fitness,runs):
    for p in range(runs):
        plt.plot(range(iterations), fitness[p], label = f"P{p}")
    plt.legend()  
    plt.show()


    
def runPSO(dataset, N):
    particles = []

    # 1 initialize each particle to contain N_c randomly selected cluster centroids
    for _ in range(NParticles):
        particles.append(Particle(dataset, N))
    globalBest = particles[0].centroids
    

    fitnesses = []
    # 2 for t=1 to t_max (max iterations) do
    for i in range(iterations):
        print(i)
        # (a) for each particle i do
        fitness_iteration = []
        for i in particles:
            # (b) for each datavector z_p
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
        globalBestParticle = particles[np.where(fitness_iteration == np.amin(fitness_iteration))[0][0]]
        globalBest = globalBestParticle.centroids
        fitnesses.append(Particle.calc_fitness_8(globalBestParticle))
        # (d) update cluster centroids using velocity and position update rules
        for p in particles:
            Particle.update_velocity(p, globalBest)
            Particle.update_centroids(p)
            Particle.reset_clusters(p)
    return fitnesses

# ======= MAIN =======================================
runs = 30
#data = gen_dataset_1()
data = gen_dataset_2("iris.data")
k_means = KMeans(10)

fitness1 = [[] for _ in range(runs)]
for i in range(runs):
    print(f"PSOrun {i}")
    fitness1[i] = runPSO(data, 4)
plot(fitness1, runs)


fitness2 = [[] for _ in range(runs)]
for i in range(runs):
    print(f"run {i}")
    fitness2[i] = k_means.fit(data)
print(fitness2)
plot(fitness2, runs)