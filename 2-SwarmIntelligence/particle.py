import numpy as np
import random

""""
Particle is a position in a N dimensional space.
"""
class Particle:
    
    def __init__(self, data, amount_classes):
        self.centroids = random.sample(data, amount_classes)
  
        self.velocity = 0
        self.current_pos = self.centroids
        self.best_pos = self.centroids
        self.clusters = np.zeros((len(data),amount_classes))

    def update(self, global_best_pos, data):
        return 0

    def update_velocity(self):
        return 0

    def update_centroids(self):
        return 0



def quantization_error():
    return 0