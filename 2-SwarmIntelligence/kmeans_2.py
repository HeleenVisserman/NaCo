    def update_centroids_kmeans(self):
        for i in range(len(self.centroids)):
            cluster = self.clusters[i].cluster
            self.centroids[i] = 1/(len(cluster)) * cluster.sum(axis=0)