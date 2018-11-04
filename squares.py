import numpy as np
from sklearn.cluster import KMeans
import sys
from itertools import product
import matplotlib.pyplot as plt

def processArgs():
    args = sys.argv
    suppliedArgs = len(args) - 1
    requiredArgs = 5
    if (suppliedArgs != requiredArgs):
        print('Need 5 args: nClustersPerDim, nSize, clusterSepartionByClusterSize, resultsDir, errorFactor ... Exiting')
        sys.exit(0)
    else:
        nClustersPerDim = int(args[1])       # 2
        nSize = int(args[2])            # 1000
        clusterSepartionByClusterSize = float(args[3])            # 4.0
        resultsDir = str(args[4])            # 4.0
        errorFactor = float(args[5])            # 4.0
    return nClustersPerDim, nSize, clusterSepartionByClusterSize, resultsDir, errorFactor

def getCoordinates (nClustersPerDim, nSize, clusterSepartionByClusterSize):
    print ('nClusters Per Dim, nSize, clusterSepartion/ClusterSize:',nClustersPerDim, nSize, clusterSepartionByClusterSize)
    intraClusterSize = 1.0 / ( nClustersPerDim*(clusterSepartionByClusterSize+1) - clusterSepartionByClusterSize)
    interClusterSize = intraClusterSize * clusterSepartionByClusterSize
    print ('intraClusterSize interClusterSize:',intraClusterSize, interClusterSize)
    xMin = 0.0
    lines = []
    coords = []
    for i in range (0, nClustersPerDim):
        xMax = xMin + intraClusterSize
        dx = intraClusterSize / nSize
        lines.append(xMax + dx)
        xl = np.linspace (xMin+2*dx, xMax-2*dx, num=nSize)
        coords.append(xl.tolist())
        xMin = xMax + interClusterSize
    coords = [item for sublist in coords for item in sublist]
    lines.pop()
    return intraClusterSize, interClusterSize, coords, lines

def getMinClusterSeparation (guessNclusters, cluster_centers):
    minClusterSeparation = 1.0e32
    for i in range (0, guessNclusters):
        for j in range (0, guessNclusters):
            if (i != j):
                diff = cluster_centers[i] - cluster_centers[j]
                dist = np.dot(diff, diff)
                minClusterSeparation = min(minClusterSeparation, dist)
    return minClusterSeparation

def plotClusters (X, predictedLabels, nPredictedClusters, clusterCenters, resultsDir, lines):
    fig = plt.figure(figsize=(6,6),dpi=720)
    xmax = 1.0
    ymax = 1.0
    subplot = fig.add_subplot(1, 1, 1)
    subplot.set_xlim (0.0, xmax)
    subplot.set_ylim (0.0, ymax)

    colors = []
    colors0 = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'DarkViolet', 'Gold', 'DarkCyan', 'Black', 'LightGrey', 'Chocolate', 'CadetBlue', 'FireBrick', 'Peru', 'Yellow', 'Indigo', 'Bisque', 'Tan', 'Wheat', 'Fuchsia', 'Plum', 'OliveDrab', 'Maroon']
    for i in range (0, 5000):
        colors = colors + colors0

    for cluster in range(0, nPredictedClusters):
        indices = np.where(predictedLabels == cluster)
        centerX = clusterCenters[cluster, 0]
        centerY = clusterCenters[cluster, 1]
        subplot.plot(X[indices,0], X[indices,1], color=colors[cluster], markersize=3, marker='*', linestyle='')
        subplot.text(centerX, centerY, cluster, fontweight='bold',fontsize=8,color='b', horizontalalignment='center', verticalalignment='center')

    zero2one = np.linspace (0, 1.0, num=10)
    ones = np.ones(10,)
    for value in lines:
        subplot.plot(zero2one, value*ones, color='k', linewidth=2)
        subplot.plot(value*ones, zero2one, color='k', linewidth=2)

    fig.savefig(resultsDir + '/pred-clusters-' + str(nPredictedClusters) + '.png', format='png', dpi=720)
    plt.close(fig)

def runKmeans (X, guess, resultsDir, lines):
    kmeans = KMeans(n_clusters=guess, max_iter=1000, tol=1.0e-6)
    labels = kmeans.fit_predict (X)
    if (guess < 100):
        plotClusters (X, labels, guess, kmeans.cluster_centers_, resultsDir, lines)
    minClusterSeparation = getMinClusterSeparation (guess, kmeans.cluster_centers_)
    ratio = kmeans.inertia_ / minClusterSeparation
    print (guess, kmeans.inertia_, minClusterSeparation,ratio)

def main():
    nClustersPerDim, nSize, clusterSepartionByClusterSize, resultsDir, errorFactor = processArgs()
    intraclusterSize, interClusterSize, coords, lines = getCoordinates (nClustersPerDim, nSize, clusterSepartionByClusterSize)

    X = []
    for xloc, yloc in product(coords, coords):
        errorX = errorFactor * np.random.randint(low=-1,high=2) * intraclusterSize * np.random.random_sample() / nSize
        errorY = errorFactor * np.random.randint(low=-1,high=2) * intraclusterSize * np.random.random_sample() / nSize
        X.append([xloc + errorX, yloc + errorY])
    X = np.array([np.array(xi) for xi in X])
    X = X.reshape(-1,2)

    maxClusters = nSize*nSize*nClustersPerDim*nClustersPerDim + 1

    print ('nClusters, Inertia, minClusterSeparation, Inertia/minClusterSeparation')
    for guess in range(2, min(maxClusters, 101)):
        runKmeans(X, guess, resultsDir, lines)
    for guess in range(101, min(maxClusters,201), 10):
        runKmeans(X, guess, resultsDir, lines)
    for guess in range(201, min(maxClusters, 1001), 100):
        runKmeans(X, guess, resultsDir, lines)
    for guess in range(1001, maxClusters, 1000):
        runKmeans(X, guess, resultsDir, lines)

    runKmeans(X, nSize*nSize*nClustersPerDim*nClustersPerDim, resultsDir, lines)

if __name__ == '__main__':
    np.set_printoptions(linewidth=100)
    main()

