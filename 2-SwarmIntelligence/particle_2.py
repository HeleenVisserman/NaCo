import math
from cv2 import sumElems
import numpy as np
import random

""""
Cluster is a list of datapoints that belong to a certain cluster
"""
class Cluster:
    def __init__(self):
        self.cluster = np.array([])
       

""""
Particle is a list of cluster centroids that is being optimized
"""
class Particle:
    
    def __init__(self, data, amount_classes):
        self.centroids = np.array(random.sample(list(data), amount_classes))
  
        self.velocity = np.zeros(amount_classes)
        self.current_pos = self.centroids
        self.local_best = self.centroids
        self.clusters = [Cluster() for c in self.centroids]

        self.w = 0.72
        self.c1 = 1.49
        self.c2 = 1.49
        self.r1 = 1.0
        self.r2 = 1.0
        # Indeed every particle has list of cluster and within this particle you update the location of the clusters based on their fit to the datapoints (this is the local best). 
        # Then you can compare the outcomes of the particles and find the global best. 
        # So you should indeed calculate for all datapoints their distance to the centroids in a particle, and do this for all particle. 

    def update_velocity(self, global_best):
        self.r1 = random.uniform(0,1)
        self.r2 = random.uniform(0,1)
        return self.w*self.velocity# + self.c1*self.r1*(self.local_best - self.current_pos) #+ self.c2*self.r2*(global_best - self.current_pos)

    def update_current_pos(self):
        return self.current_pos + self.velocity

    
    
    def update_local_best(self, new_pos, old_local_best):
        return 0

    def update_centroids(self):
        return 0
    
    def calc_euclidean_distance(self, c, z):
        return math.sqrt(sum((c-z)**2))

    def calc_fitness_8(self):
        sum = 0
        for c in range(len(self.centroids)):
            cluster = self.clusters[c].cluster
            for z in cluster:
                sum += (self.calc_euclidean_distance(self.centroids[c], z) / len(cluster))
        return sum/len(self.centroids)

# ======= MAIN =======================================

iterations = 100
NParticles = 10

def gen_dataset_1():
    vectors = np.array([])
    for _ in range(400):
        z1 = random.uniform(-1, 1)
        z2 = random.uniform(-1, 1)
        
        vectors.append(np.array([z1, z2]))
    return vectors   
    
def runParticles(dataset, N):
    data = dataset

    # 1 initialize each particle to contain N_c randomly selected cluster centroids
    particles = np.array([Particle(dataset, N)]*NParticles)

    fitnesses = []*iterations
    # 2 for t=1 to t_max (max iterations) do
    for i in range(iterations):
        # (a) for each particle i do
        fitness_iteration = []
        for i in particles:
            # (b) for each datavector z_p
            for z in data:
                # i calculate the Euclidean distance d(z_p, m_ij) to all cluster centroids c_ij
                distance = None
                cluster = None
                for j in range(len(i.centroids)):
                    c = i.centroids[j]
                    new_distance = Particle.calc_euclidean_distance(c,z)
                    if distance is None or distance > new_distance:
                        distance = new_distance
                        cluster = j
                
                # ii assign z_p to cluster C_ij with minimal distance
                i.clusters[j].append(z)
            # iii calculate fitness using equation (8)
            fitness_iteration.append(Particle.calc_fitness_8(i))
        fitnesses[i] = fitness_iteration
        


if __name__ == "__main__":
    # run(gen_dataset_1(),2)
    # run(gen_dataset_2("iris.data"),3)

data = np.array([[1,2,3], [2,3,4],[43,5,2],[32,4,56]])
a = Particle(data, 2)
print(a.centroids)
m = data.sum(axis=0)
Particle.update_centroids(a)
print(a.centroids)

e = a.calc_euclidean_distance(a.centroids[0],data[0])
print(e)

f = Particle.calc_fitness_8(a)
print(f)

print(2*3*(a.local_best + a.current_pos))
Particle.update_velocity(a, a.centroids)
print(a.velocity)

def quantization_error():
    return 0
 