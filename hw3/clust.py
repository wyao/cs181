# clust.py
# -------
# Irene Chen and Willie Yao

import sys
import random
from utils import *
import operator
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
    if(validateInput() == False):
        print "Usage: clust numClusters numExamples"
        sys.exit(1);

    numClusters = int(sys.argv[1]) #K
    numExamples = int(sys.argv[2]) #n

    #Initialize the random seed
    
    random.seed()

    #Initialize the data

    
    dataset = file(DATAFILE, "r")
    if dataset == None:
        print "Unable to open data file"


    data = parseInput(dataset)
    
    
    dataset.close()
    #printOutput(data,numExamples)
    assert(numExamples <= len(data))
    assert(numClusters <= numExamples)

    # ==================== #
    # WRITE YOUR CODE HERE #
    # ==================== #
    #py.test.set_trace()
    # K-means
    dimension = len(data[0])
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
        if newAssignments == assignments:
            converged = True
        assignments = newAssignments

        # Update prototype vectors
        for k in xrange(numClusters):
            prototypes[k] = [0.]*dimension
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

if __name__ == "__main__":
    validateInput()
    main()
