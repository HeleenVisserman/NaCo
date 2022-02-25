class KMeans:
    """"K-Means clustering algorithm."""
    def __init__(
        self,
        k: int):
        self.k = k

    def fit(self):
        #t
        return 0;
        

    
def update_centroids_kmeans(self):
    for i in range(len(self.centroids)):
        cluster = self.clusters[i].cluster
        self.centroids[i] = 1/(len(cluster)) * cluster.sum(axis=0)