import algorithms.kmeans as kmeans
import algorithms.minibatch_kmeans as minibatch_kmeans
import algorithms.spectral as spectral
import algorithms.meanshift as meanshift
import algorithms.agglomerative as agglomerative
import algorithms.affinity as affinity
import algorithms.dbscan as dbscan
import algorithms.optics as optics

import numpy as np
import matplotlib.pyplot as plt

import sys

PLOT_N = 4
PLOT_M = 2
CURR_PLOT = 1

def print_data(data, labels):
    print("Printing entries with their labels...")

    for i in range(len(data)):
        print("entry {0} with label {1}".format(data[i], labels[i]))

def print_result(labels, centers):
    print("Labels:\n%s" % labels)
    if centers is not None:
        print("Centers:")
        for center in centers:
            print(center)

def plot_results(data, labels, name):
    global CURR_PLOT
    print("Plotting {0} with first two dimensions...".format(name))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for label in list(set(labels)):
        plt.subplot(PLOT_N, PLOT_M, CURR_PLOT)
        plt.plot(
            [data[x][0] for x in range(data.shape[0]) if labels[x] == label],
            [data[x][1] for x in range(data.shape[0]) if labels[x] == label],
            'o' + colors[label % len(colors)],
            label="cluster #{0}".format(label)
        )
        plt.title(name)
    CURR_PLOT += 1
    print("Plot added.\n")

def test_kmeans(n_clusters, data):
    print("Testing kmeans with k={0}".format(n_clusters))
    labels, centers = kmeans.fit(n_clusters, data)
    print_result(labels, centers)
    plot_results(data, labels, "kmeans")

def test_mbkmeans(n_clusters, data):
    print("Testing minibatch kmeans with k={0}".format(n_clusters))
    labels, centers = minibatch_kmeans.fit(n_clusters, data)
    print_result(labels, centers)
    plot_results(data, labels, "mini batch kmeans")

def test_spectral(n_clusters, data):
    print("Testing spectral clustering with k={0}".format(n_clusters))
    labels, centers = spectral.fit(n_clusters, data)
    print_result(labels, centers)
    plot_results(data, labels, "spectral")

def test_meanshift(n_clusters, data):
    print("Testing mean shift")
    labels, centers = meanshift.fit(n_clusters, data)
    print("{0} clusters obtained".format(len(set(labels))))
    print_result(labels, centers)
    plot_results(data, labels, "mean shift")

def test_agglomerative(n_clusters, data):
    print("Testing agglomerative")
    labels, centers = agglomerative.fit(n_clusters, data)
    print("{0} clusters obtained".format(len(set(labels))))
    print_result(labels, centers)
    plot_results(data, labels, "agglomerative")

def test_affinity(n_clusters, data):
    print("Testing affinity")
    labels, centers = affinity.fit(n_clusters, data)
    print("{0} clusters obtained".format(len(set(labels))))
    print_result(labels, centers)
    plot_results(data, labels, "affinity")

def test_dbscan(n_clusters, data):
    print("Testing dbscan")
    labels, centers = dbscan.fit(n_clusters, data)
    print("{0} clusters obtained".format(len(set(labels))))
    print_result(labels, centers)
    plot_results(data, labels, "dbscan")

def test_optics(n_clusters, data):
    print("Testing optics")
    labels, centers = optics.fit(n_clusters, data)
    print("{0} clusters obtained".format(len(set(labels))))
    print_result(labels, centers)
    plot_results(data, labels, "optics")

def main():
    if len(sys.argv) < 3:
        print("Usage:\tpython {0} <path to file> <n clusters>".format(sys.argv[0]))
        return

    filename = sys.argv[1]
    n_clusters = int(sys.argv[2])

    data = np.genfromtxt(filename, delimiter=",")
    print("Loaded {0} records with {1} features each from {2} file\n".format(
        data.shape[0], data.shape[1], filename
    ))

    test_kmeans(n_clusters, data.copy())
    test_mbkmeans(n_clusters, data.copy())
    test_spectral(n_clusters, data.copy())
    test_meanshift(n_clusters, data.copy())
    test_affinity(n_clusters, data.copy())
    test_agglomerative(n_clusters, data.copy())
    test_dbscan(n_clusters, data.copy())
    test_optics(n_clusters, data.copy())

    plt.show()

if __name__ == "__main__":
    main()