import numpy as np
import random
import math

class KMeans:
    """"K-Means clustering algorithm."""
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

    def fit(self, data: np.array):
        self._init_centroids(data)
        for _ in range(self.max_iterations):
            for v in data:
                self._calculate_distances(v)
            total_distance = self._calculate_sse()
            if self.best_total_distance is None or self.best_total_distance > total_distance:
                self.best_total_distance = total_distance
                self.best_clusters = self.clusters
                self.best_centroids = self.centroids
            
            self._calculate_centroids()
            self.clusters = [[] for _ in range(self.k)]


        return self.best_clusters

    def _calculate_centroids(self):
        for c in range(self.k):
            centroid = np.mean(self.clusters[c], axis=0) # [ [1,2,3,4],[2,3,4,5] ]
            #sum = sum(np.array(self.clusters[c]))
            self.centroids[c] = centroid

    def _init_centroids(self, data: np.array):
        """Initialize centroids."""
        #self.centroids = random.sample(data, self.k)
        self.centroids = np.array(random.sample(list(data), self.k))
        # randomly initialize the N_c cluster centroid vectors 
        # for _ in range(self.k):
            # self.centroids.append(random.choice(data, size=self.k)
        
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
            for d in self.clusters[c]:
                total_distance += self.calc_euclidean_distance(d,self.centroids[c])

        return total_distance

    
    
def update_centroids_kmeans(self):
    for i in range(len(self.centroids)):
        cluster = self.clusters[i].cluster
        self.centroids[i] = 1/(len(cluster)) * cluster.sum(axis=0)


# ======= MAIN =======================================
data = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
k_means = KMeans(2)

clusters = k_means.fit(data)
print(clusters)