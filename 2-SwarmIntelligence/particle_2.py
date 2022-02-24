import math
from cv2 import sumElems
import numpy as np
import random

iterations = 100
NParticles = 10
w = 0.72
c1 = 1.49
c2=1.49

""""
Cluster is a list of datapoints that belong to a certain cluster
"""
class Cluster:
    def __init__(self):
        self.cluster = np.array([[1,2,3],[32,4,56]])
       

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
 