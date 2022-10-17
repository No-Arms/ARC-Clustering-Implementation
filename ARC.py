'''
Adaptive Radius Clustering Algorithm Implementation

CSCI4144 - Data Mining/Warehousing: Final Project Code
Group 26
Robert Carter, B00801408
Noah Armsworthy, B00324836
Conor French, B00805700

April 10, 2022
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn import metrics
import copy
from scipy.special import comb


class ARC():

    def __init__(self, dataset, eFunc):
        self.eFunc = eFunc
        self.dataset = dataset
        self.distanceMatrix = self.datasetDistance(dataset)
        self.totalDistance = np.sum(self.distanceMatrix)
        self.clusters = []
        self.orderedLabels = None

    # Returns labels provided in same order as datapoints. Valuable for comparisons.
    def getOrderedLabels(self):

        if not(isinstance(self.orderedLabels, np.ndarray)):
            self.orderedLabels = np.zeros(len(self.dataset)) - 1
            for i in range(len(self.clusters)):
                for j in range(len(self.clusters[i])):
                    self.orderedLabels[self.clusters[i][j]] = i

        return self.orderedLabels

    # cluster is the main function for taking the dataset and assigning clusters.
    def cluster(self):

        # Set all datapoints as unclustered.
        unclustered = np.arange(len(self.dataset))

        # while unclustered is non empty, keep going.
        while unclustered.any(): 

            # Define v1 as the least different point to all other points.
            v1 = self.leastDistantPoint(self.dataset[unclustered])

            # Compute v2, v3 only if dataset has 3 or more points remaining.
            if len(unclustered) >= 3:
                v2 = self.Equation56(self.dataset[unclustered], v1)
                v3 = self.Equation56(self.dataset[unclustered], v1, v2)

            # Check for bad initialization and if v1, v3 exist.
            if len(unclustered) >= 3 and self.Equation7(self.dataset[unclustered], v1, v3, self.totalDistance):
                
                # Initialize current cluster.
                currCluster = [unclustered[v1], 
                                unclustered[v2], 
                                unclustered[v3]]

                mask = np.ones_like(unclustered, dtype=bool)
                mask[[v1, v2, v3]] = False
                unclustered = unclustered[mask]
                frontier = list(copy.deepcopy(currCluster))
                
                # For whatever reason, v1 is NOT used to extend the cluster. So removing it this way.
                frontier.pop(0)

                # Initializing the historical velocities.
                distanceHistory = self.distanceMatrix[currCluster,:][:,currCluster].flatten()
                distanceHistory = distanceHistory[distanceHistory != 0]

                # check for neighbors 
                while bool(frontier) and len(unclustered) > 0:

                    # Step 2
                    while bool(frontier):
                        # Find epsilon neighbors of current point.
                        
                        currPoint = frontier[0]

                        # Estimating mean and standard deviation of dataset.
                        mean = np.mean(distanceHistory)
                        stdev = np.std(distanceHistory, ddof=1)
                        
                        epsilon = self.eFunc(mean, stdev, len(distanceHistory))
                        #print(epsilon)
                        
                        # newPoints is a boolean mask intended for the unclustered data, if True, then a point is a neihgbor and added to cluster.
                        newPoints = self.neighbors(self.distanceMatrix, unclustered, currPoint, epsilon)
                        distanceHistory = np.append(distanceHistory, self.distanceMatrix[currPoint][unclustered[newPoints]])
                        
                        # Add any new points to the frontier and to the current cluster.
                        frontier = frontier + list(unclustered[newPoints])
                        currCluster = currCluster + list(unclustered[newPoints]) 

                        # Remove newly clustered points from the unclustered list.
                        unclustered = unclustered[~newPoints]
            
                        # Remove the point we just checked for neighbors from the frontier.
                        frontier.pop(0)

                #print(distanceHistory)

            # Only v1 is in this cluster -> total outlier.
            else:
                currCluster = [unclustered[v1]]    
                mask = np.ones_like(unclustered, dtype=bool)
                mask[v1] = False
                unclustered = unclustered[mask]

            # Add current cluster to completed clusters.
            self.clusters.append(list(currCluster))


    # Return pairwise distances between every pair of datapoints.
    def datasetDistance(self, dataset):
    
        distMatrix = np.zeros((len(dataset), len(dataset)))
    
        # For big and high dimension data
        if len(dataset) > 1000 & len(dataset[0] > 5):
            for i in range(len(dataset[0])):
              distMatrix += (dataset[:, None, i] - dataset[None, :, i]) * (dataset[:, None, i] - dataset[None, :, i])

            distMatrix = np.sqrt(distMatrix)
    
            return distMatrix
    
        else:
    
            return np.linalg.norm(dataset[:, None, :] - dataset[None, :, :], 
            axis=-1)

    # Euclidean distance between two points.
    def distance(self, v1, v2):
        return np.sum((v1 - v2)**2)**(1/2)

    # Out of entire dataset, get least distant point. Equivalent to equation 4
    def leastDistantPoint(self, dataset):
        return np.argmin(np.sum(self.datasetDistance(dataset),axis=0))

    # Equation 5/6: distance from v1
    def Equation56(self, dataset, v1, v2=False):
        
        counter = 1
        if v2:
            counter = 2

        point = np.reshape(dataset[v1], [1, -1])

        return np.argpartition(np.linalg.norm(dataset[:, None, :] - point[None, :, :], 
        axis=-1).reshape(-1), counter)[counter]

    # One option for computing epsilon
    def Equation2(sampleMean, stdev, n):
        return sampleMean + 2*stdev

    # Another option for computing epsilon.
    def Equation3(sampleMean, stdev, n):
        return sampleMean + ((1.96 * stdev) / (np.sqrt(n)))

    # Condition for deciding if we have a bad initialization.
    def Equation7(self, dataset, v1, v3, dist):
        return (self.distance(dataset[v3], dataset[v1])
        < 1 / comb(len(dataset), 2) * dist)

    # Returns epsilon-radius neighbors.
    # dataset should be an np.array of points, centerPoint an index for it.
    def neighbors(self, distanceMatrix, otherPoints, centerPoint, epsilon):

        # True for each point that is an epsilon neighbor.
        # False for each point that isn't.
        mask = self.distanceMatrix[centerPoint][otherPoints] <= epsilon#np.linalg.norm(dataset[:, None, :] - point[None, :, :], 
        # axis=-1) < epsilon
        #print(mask == np.linalg.norm(dataset[:, None, :] - point[None, :, :], axis=-1) < epsilon)
        mask = np.reshape(mask, -1)
        return mask


def dataFromText(fp):
    data = np.genfromtxt(fp, delimiter='	', dtype=float)
    labels = data[:, -1]
    X = data[:, :-1]
    maxValues = np.max(X, axis=0)
    minValues = np.min(X, axis=0)
    X = (X - minValues) / (maxValues - minValues)
    return X, labels


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    spirX, spirL = dataFromText('datasets/spiral.txt')
    aggX, aggL = dataFromText('datasets/aggregation.txt')
    compX, compL = dataFromText('datasets/compound.txt')

    data = [spirX, aggX, compX]
    dataNames = ['Spiral','Aggregation', 'Compound']
    labels = [spirL, aggL, compL]
    arcL = []
    equations = [ARC.Equation2, ARC.Equation3]
    #0 or 1 based on which epsilon equation to run
    pickedEq = 1

    for dataset in data:
        arc = ARC(dataset, equations[pickedEq])
        arc.cluster()
        arcL.append(arc.getOrderedLabels())

    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    for i in range(0, len(axs)):
        axs[i].scatter(data[i][:, 0], data[i][:, 1], c=arcL[i], cmap='viridis')
        axs[i].set_title(dataNames[i])
    fig.suptitle(f'ARC results using Epsilon Equation {pickedEq+1}')
    fig.tight_layout()
    plt.subplots_adjust(top=0.85)
    fig.savefig(f'ArcEq{pickedEq+1}Plots')
    plt.show()

    #Exporting score results to .txt file
    title = "ARC Adjusted Rand Index Scores:"
    print(title, file=open("Scores.txt", "w"))
    print('-'*len(title), file=open("Scores.txt", "a"))
    for i in range(0, len(labels)):
        score = metrics.adjusted_rand_score(labels[i], arcL[i])
        print(f"{dataNames[i]}: {round(score,4)}", file=open("Scores.txt", "a"))

    #Exporting label results to .csv files
    for i in range(0, len(data)):
        df = pandas.DataFrame((np.vstack((arcL[i], labels[i]))).T, columns=['Predicted','True'], dtype=np.float64)
        fileName = dataNames[i] + 'Results.csv'
        df.to_csv(fileName, index=False)






