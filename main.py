from KMeans import KMeans
import matplotlib.pyplot as plt
from sklearn import datasets

if __name__ == "__main__":

    #data_t = [[167,55],[120,32],[113,33],[175,76],[108,25]]
    data_t = [
        [1.9, 7.3],
        [3.4, 7.5], 
        [2.5, 6.8], 
        [1.5, 6.5],
        [3.5, 6.4],
        [2.2, 5.8],
        [3.4, 5.2],
        [3.6, 4.0],
        [5, 3.2],
        [4.5, 2.4],
        [5, 3.2],
        [4.5, 2.4],
        [6, 2.6],
        [1.9, 3],
        [1, 2.7],
        [1.9, 3],
        [1, 2.7],
        [1.9, 2.4],
        [0.8, 2],
        [1.6, 1.8],
        [1, 1]
        ]


    print("\n** Exercise 1 Dataset**")

    km = KMeans(3, data_t)

    km.fit()

    print("Centroids: \n")
    km.print_centroids()

    plt.scatter(x=[v[0] for v in km.centroids], y=[v[1] for v in km.centroids], c=['red', 'blue', 'green'])
    plt.scatter(x=[d[0] for d in data_t], y=[d[1] for d in data_t])
    plt.title("K-Means Exercise 1")

#######

    print("\n ** Iris Dataset **")

    data_iris = datasets.load_iris(return_X_y=True)[0]

    print("\n K=3")
    km_iris = KMeans(3, data_iris)

    print("Centroids: \n")
    km_iris.fit()
    km_iris.print_centroids()

#######

    print("\n ** Wine Dataset **")

    data_wine = datasets.load_wine(return_X_y=True)[0]
    
    k = 5
    print("\n K={}".format(k))
    km_wine = KMeans(k, data_wine)

    print("Centroids: \n")
    km_wine.fit()
    km_wine.print_centroids()

#######

    print("\n ** Breast Cancer Dataset **")

    data_cancer = datasets.load_breast_cancer(return_X_y=True)[0]
    
    k = 2
    print("\n K={}".format(k))
    km_cancer = KMeans(k, data_cancer)

    print("Centroids: \n")
    km_cancer.fit()
    km_cancer.print_centroids()

    plt.show()