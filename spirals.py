import numpy as np
from sklearn.cluster import KMeans
import sys
from itertools import product
import matplotlib.pyplot as plt

def processArgs():
    args = sys.argv
    suppliedArgs = len(args) - 1
    requiredArgs = 3
    if (suppliedArgs != requiredArgs):
        print('Need 3 args: nClustersPerDim, nSize, resultsDir ... Exiting')
        sys.exit(0)
    else:
        nClustersPerDim = int(args[1])       # 2
        nSize = int(args[2])            # 1000
        resultsDir = str(args[3])            # 4.0
    return nClustersPerDim, nSize, resultsDir

def getSpiral(dx, nSize):
    x = []
    y = []
    pi = np.pi
    maxRevolutions = 5
    for theta in np.linspace(0,2*maxRevolutions*pi, nSize):
        r = theta * (dx/2) / (2*maxRevolutions*pi)
        x.append(r*np.cos(theta))
        y.append(r*np.sin(theta))
    return x, y

def getMinClusterSeparation (guessNclusters, cluster_centers):
    minClusterSeparation = 1.0e32
    for i in range (0, guessNclusters):
        for j in range (0, guessNclusters):
            if (i != j):
                diff = cluster_centers[i] - cluster_centers[j]
                dist = np.dot(diff, diff)
                minClusterSeparation = min(minClusterSeparation, dist)
    return minClusterSeparation

def plotPerfMetrics (perfMetrics, resultsDir):
    perfMetrics = np.array([np.array(xi) for xi in perfMetrics])    #   clusters, kmeans.inertia_, minClusterSeparation, ratio
    fig = plt.figure(figsize=(6,6),dpi=720)
    subplot = fig.add_subplot(1, 1, 1)
    subplot.plot(perfMetrics[:,0], perfMetrics[:,1], color='b')
    subplot.plot(perfMetrics[:,0], perfMetrics[:,2], color='r')
    subplot.plot(perfMetrics[:,0], perfMetrics[:,3], color='g')
    fig.savefig(resultsDir + '/perfMetrics.png', format='png', dpi=720)
    plt.close(fig)

    fig = plt.figure(figsize=(6,6),dpi=720)
    subplot = fig.add_subplot(1, 1, 1)
    subplot.semilogy(perfMetrics[:,0], perfMetrics[:,1], color='b')
    subplot.semilogy(perfMetrics[:,0], perfMetrics[:,2], color='r')
    subplot.semilogy(perfMetrics[:,0], perfMetrics[:,3], color='g')
    fig.savefig(resultsDir + '/log-perfMetrics.png', format='png', dpi=720)
    plt.close(fig)

def plotClusters (X, predictedLabels, nPredictedClusters, clusterCenters, resultsDir):
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

    fig.savefig(resultsDir + '/pred-clusters-' + str(nPredictedClusters) + '.png', format='png', dpi=720)
    plt.close(fig)

def runKmeans (X, guess, resultsDir):
    kmeans = KMeans(n_clusters=guess, max_iter=1000, tol=1.0e-6)
    labels = kmeans.fit_predict (X)
    if (guess < 100):
        plotClusters (X, labels, guess, kmeans.cluster_centers_, resultsDir)
    minClusterSeparation = getMinClusterSeparation (guess, kmeans.cluster_centers_)
    ratio = kmeans.inertia_ / minClusterSeparation
    print (guess, kmeans.inertia_, minClusterSeparation,ratio)

def main():
    nClustersPerDim, nSize, resultsDir = processArgs()
    dx = 1.0/nClustersPerDim
    spx, spy = getSpiral(dx, nSize)
    locs = np.linspace (dx/2, 1.0-dx/2, nClustersPerDim)
    X = []
    for xloc, yloc in product(locs, locs):
        for i in range(0,len(spx)):
            X.append([xloc + spx[i], yloc + spy[i]])
    X = np.array([np.array(xi) for xi in X])
    X = X.reshape(-1,2)

    maxClusters = nSize*nClustersPerDim*nClustersPerDim + 1

    print ('nClusters, Inertia, minClusterSeparation, Inertia/minClusterSeparation')
    for guess in range(2, min(maxClusters, 101)):
        runKmeans(X, guess, resultsDir)
    for guess in range(101, min(maxClusters,201), 10):
        runKmeans(X, guess, resultsDir)
    for guess in range(201, min(maxClusters, 1001), 100):
        runKmeans(X, guess, resultsDir)
    for guess in range(1001, maxClusters, 1000):
        runKmeans(X, guess, resultsDir)

    runKmeans(X, nSize*nClustersPerDim*nClustersPerDim, resultsDir)

if __name__ == '__main__':
    np.set_printoptions(linewidth=100)
    main()

