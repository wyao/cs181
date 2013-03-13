# clust.py
# -------
# Irene Chen and Willie Yao

import sys
import random
from utils import *
import operator
from optparse import OptionParser
import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import py.test

DATAFILE = "adults.txt"

#validateInput()

def validateInput():
    if len(sys.argv) != 3:
        return False
    if sys.argv[1] <= 0:
        return False
    if sys.argv[2] <= 0:
        return False
    return True


#-----------


def parseInput(datafile):
    """
    params datafile: a file object, as obtained from function `open`
    returns: a list of lists

    example (typical use):
    fin = open('myfile.txt')
    data = parseInput(fin)
    fin.close()
    """
    data = []
    for line in datafile:
        instance = line.split(",")
        instance = instance[:-1]
        data.append(map(lambda x:float(x),instance))
    return data


def printOutput(data, numExamples):
    for instance in data[:numExamples]:
        print ','.join([str(x) for x in instance])

# main
# ----
# The main program loop
# You should modify this function to run your experiments

def main():
    # Validate the inputs
    # if(validateInput() == False):
    #     print "Usage: clust numClusters numExamples [-k]"
    #     sys.exit(1);

    parser = OptionParser()
    parser.add_option('-k', action='store_true',
        dest='k_means', default=False)
    parser.add_option('--min', action='store_true', default=False)
    parser.add_option('--max', action='store_true', default=False)
    parser.add_option('--mean', action='store_true', default=False)
    parser.add_option('--cent', action='store_true', default=False)
    parser.add_option('--auto', action='store_true', default=False)
    parser.add_option('--small', action='store_true', default=False)

    (opt, args) = parser.parse_args()

    numClusters = int(args[0]) #K
    numExamples = int(args[1]) #n

    #Initialize the random seed
    random.seed()

    #Initialize the data
    dataset = None
    if opt.small:
        dataset = file('adults-small.txt', 'r')
    elif opt.k_means or opt.auto:
        dataset = file(DATAFILE, "r")
    elif opt.max or opt.min or opt.mean or opt.cent:
        dataset = file('adults-small.txt', 'r')
    if dataset == None:
        print "Unable to open data file"
        exit(1)

    data = parseInput(dataset)
    
    dataset.close()
    #printOutput(data,numExamples)
    assert(numExamples <= len(data))
    assert(numClusters <= numExamples)

    # ==================== #
    # WRITE YOUR CODE HERE #
    # ==================== #

    dimensions = len(data[0])

    # K-means
    if opt.k_means:
        chosen = set()
        prototypes = []
        assignments = [] # Aka responsibility vectors

        # Initialize prototype vectors
        for _ in xrange(numClusters):
            r = None
            while True:
                r = random.randint(0, numExamples-1)
                if r not in chosen:
                    chosen.add(r)
                    break
            prototypes.append(list(data[r]))

        # Repeatedly update responsibility and prototype vectors
        converged = False
        iteration = 1
        while not converged:
            print iteration
            # Update responsibility vectors
            newAssignments = []
            for n in xrange(numExamples):
                bestProto = None
                minDistance = float('inf')
                for k in xrange(numClusters):
                    d = squareDistance(data[n], prototypes[k])
                    if d < minDistance:
                        bestProto = k
                        minDistance = d
                newAssignments.append(bestProto)
            converged = (newAssignments == assignments)
            assignments = newAssignments

            # Update prototype vectors
            for k in xrange(numClusters):
                prototypes[k] = [0.]*dimensions
            counts = [0]*numClusters
            for n in xrange(numExamples):
                k = assignments[n]
                counts[k] += 1
                prototypes[k] = map(operator.add, prototypes[k], data[n])
            for k in xrange(numClusters):
                prototypes[k] = map(lambda x: x/counts[k], prototypes[k])

            iteration += 1
            err = 0.
            for n in xrange(numExamples):
                err += squareDistance(data[n], prototypes[assignments[n]])
            print err
        print counts

    # HAC
    elif opt.max or opt.min or opt.mean or opt.cent:
        clusters = [[list(p)] for p in data[:numExamples]]

        while len(clusters) > numClusters:
            shortest, a, b = float('inf'), None, None
            for i in xrange(len(clusters)-1):
                for j in xrange(i+1, len(clusters)):
                    # Find distance
                    dist = None
                    if opt.min:
                        dist = cmin(clusters[i], clusters[j], squareDistance)
                    elif opt.max:
                        dist = cmax(clusters[i], clusters[j], squareDistance)
                    elif opt.mean:
                        dist = cmean(clusters[i], clusters[j], squareDistance)
                    else:
                        dist = ccent(clusters[i], clusters[j], squareDistance)
                    # Update shortest distance
                    if dist < shortest:
                        shortest = dist
                        a = i
                        b = j
            # Merge clusters
            clusters[a] = clusters[a] + clusters[b]
            clusters.pop(b)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for c,color in zip(clusters, ['r','b','g','black']):
            for i in xrange(len(c)):
                ax.scatter(c[i][0],c[i][1], \
                    c[i][2],c=color)
        plt.show()

    # Autoclass
    elif opt.auto:
        # Set up discretization info
        assert(dimensions == len(attributes))

        E = {} # Expected values
        P = {} # Parameters
        mean = {}
        var = {}
        pdf = {} # Probabilities from using prob distr function

        # Set initial parameters
        for k in xrange(numClusters):
            P[k] = random.random()
            for d in xrange(dimensions):
                if attributes[d]['isContinuous']:
                    mean[(k,d)] = float(data[0][d]) # Use first dataset to seed
                    var[(k,d)] = float(data[0][d]) / 10. if data[0][d] \
                        else 1.
                else:
                    P[(k,d)] = random.random()
        # Run EM
        converged = False
        for _ in xrange(30):
            # Calculate expected values step
            # Set all expected values to 0
            for k in xrange(numClusters):
                E[k] = 0.
                for d in xrange(dimensions):
                    E[(k,d)] = 0.
            # For each input
            for n,x in enumerate(data[:numExamples]):
                # Per input temporary probabilities for each cluster
                prob = []
                for _ in xrange(numClusters):
                    prob.append(float(P[k]))
                # For each attribute calculate probabilities
                for d,xi in enumerate(x):
                    for k in xrange(numClusters):
                        if attributes[d]['isContinuous']:
                            pdf[(k,d,n)] = getPDF(float(xi), var[(k,d)], \
                                mean[(k,d)])
                            prob[k] *= pdf[(k,d,n)]
                        else:
                            if xi > 0:
                                prob[k] *= P[(k,d)]
                            else:
                                prob[k] *= (1-P[(k,d)])
                # Sum up expected values
                totalProb = sum(prob)
                for k in xrange(numClusters):
                    E[k] += prob[k]/totalProb
                    for d in xrange(dimensions):
                        if x[d] > 0:
                            E[(k,d)] += prob[k]/totalProb
            # Maximization step
            for k in xrange(numClusters):
                # Update parameters for Bernoulli's
                P[k] = E[k]/numExamples
                for d in xrange(dimensions):
                    if not attributes[d]['isContinuous']:
                        P[(k,d)] = E[(k,d)]/E[k]
                    else:
                        # Update parameters Gaussians
                        denominator = 0.
                        meanNumerator = 0.
                        varNumerator = 0.
                        for n in xrange(numExamples):
                            denominator += pdf[(k,d,n)]
                            meanNumerator += pdf[(k,d,n)] * data[n][d]
                        mean[(k,d)] = meanNumerator/denominator
                        for n in xrange(numExamples):
                            varNumerator += pdf[(k,d,n)] * \
                                math.pow(data[n][d] - mean[(k,d)],2)
                        var[(k,d)] = varNumerator/denominator
            print [P[k] for k in xrange(numClusters)]

if __name__ == "__main__":
    validateInput()
    main()
