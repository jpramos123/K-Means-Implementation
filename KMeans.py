from random import randint
import numpy as np
np.seterr('raise')

class KMeans:

    """
        This class represents the K-Means algorithm,
        create to identify clusters of data.

    """

    def __init__(self, n_centroids, data):
        self.n_centroids = n_centroids
        self.data = np.array(data)
        self.centroids = []
    
    
    def fit(self):
        """
            This method wraps other methos that are used to train the model,
            and identify the correct centroids.
        """
        self.initialize()
        self.iterate()

    def initialize(self):
        """
            This methods is used to initialize the centroids
            of the model.
        """
        centroids = [self.data[randint(0, len(self.data)-1)] for i in range(self.n_centroids)]
        self.centroids = np.array(centroids)
    
    def iterate(self):
        change = 0
        while(change != 1):

            centroids = self.find_centroids()

            if(self.check_centroid(centroids)):
                change = 1
            else:
                self.centroids = centroids

    def check_centroid(self, centroids):

        if np.array_equal(centroids, self.centroids):
            return True
        else:
            False


    def find_centroids(self):
        centroids_dist = self.distance()

        new_centroids = []
        for i in range(len(centroids_dist)):
            centroid = (np.sum(centroids_dist[i], axis=0)/len(centroids_dist[i]))
            new_centroids.append(centroid)
        
        return new_centroids

    def distance(self):
        
        centroid_dists = [[] for i in range(self.n_centroids)]

        for i in range(len(self.data)):
            b_dist = -1
            centroid_idx = -1
            for j in range(len(self.centroids)):     
                dist  = np.linalg.norm(self.centroids[j] - self.data[i])
                if b_dist == -1 or dist < b_dist:
                    b_dist = dist
                    centroid_idx = j
            centroid_dists[centroid_idx].append(self.data[i])
        return centroid_dists

    def print_centroids(self):

        for i in range(len(self.centroids)):
            print("\nC{}: \n".format(i), self.centroids[i])