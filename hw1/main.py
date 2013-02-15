# main.py
# -------
# Irene Chen and Willie Yao

from dtree import *
from pylab import *

import math
import matplotlib.pyplot as plt
import py.test
import sys

class Globals:
    noisyFlag = False
    pruneFlag = False
    valSetSize = 0
    dataset = None


##Classify
#---------

def classify(decisionTree, example):
    return decisionTree.predict(example)

##Learn
#-------
def learn(dataset):
    learner = DecisionTreeLearner()
    learner.train(dataset)
    return learner.dt

# main
# ----
# The main program loop
# You should modify this function to run your experiments

def parseArgs(args):
  """Parses arguments vector, looking for switches of the form -key {optional value}.
  For example:
    parseArgs([ 'main.py', '-n', '-p', 5 ]) = { '-n':True, '-p':5 }"""
  args_map = {}
  curkey = None
  for i in xrange(1, len(args)):
    if args[i][0] == '-':
      args_map[args[i]] = True
      curkey = args[i]
    else:
      assert curkey
      args_map[curkey] = args[i]
      curkey = None
  return args_map

def validateInput(args):
    args_map = parseArgs(args)
    valSetSize = 0
    noisyFlag = False
    pruneFlag = False
    boostRounds = -1
    maxDepth = -1
    if '-n' in args_map:
      noisyFlag = True
    if '-p' in args_map:
      pruneFlag = True
      valSetSize = int(args_map['-p'])
    if '-d' in args_map:
      maxDepth = int(args_map['-d'])
    if '-b' in args_map:
      boostRounds = int(args_map['-b'])
    return [noisyFlag, pruneFlag, valSetSize, maxDepth, boostRounds]

# Helper functions
NODE, LEAF = range(2)

def accuracy(dt, examples):
  """ returns correct/total """
  correct = 0.
  size = len(examples)

  # Count up correct predictions given target attribute
  for i in xrange(size):
    if dt.predict(examples[i]) == examples[i].attrs[-1]:
      correct += 1
  return correct/size

def getClassification(dt):
  """ Returns majority classification, or None if pruning not necessary.
      Defaults to positive.
  """
  labelCount = [0,0]
  for attr in dt.branches:
    if dt.branches[attr].nodetype is NODE:
      return None
    labelCount[dt.branches[attr].classification] += 1
  return 0 if labelCount[0] > labelCount[1] else 1


def prune(dt, root, examples):
  """ Assumptions:
        Pruning terminates as soon as a node rejects pruning.
        Labels must be binary.
        This implementation cannot prune a tree that has a depth < 2.
  """
  if dt.nodetype == LEAF:
    return dt

  for attr in dt.branches:
    # Ignore leaves
    if dt.branches[attr].nodetype is NODE:
      # Bottom-up pruning
      dt.branches[attr] = prune(dt.branches[attr], root, examples)

      # Prune from 2 layers up (if applicable)
      classification = getClassification(dt.branches[attr])
      if classification is not None:
        # Test performance without pruning
        originalPerformance = accuracy(root, examples)
        originalBranch = dt.branches[attr]

        # Test performance with pruning
        dt.branches[attr] = DecisionTree(LEAF, classification=classification)
        if accuracy(root, examples) < originalPerformance:
          dt.branches[attr] = originalBranch
  return dt

###########################################################################

def calc_error(dt, examples):
    "calculate the weighted error of a dt"
    error = 0.
    size = len(examples)

    # sum errors weights that dt classifies incorrectly
    for i in xrange(size):
        if examples[i].attrs[9] != classify(dt, examples[i]):
            error += examples[i].weight
    return error/size # nomalize to distribution

def alpha(error):
  if error==0:
    return 100.
  else:
    return 0.5 * math.log((1.-error)/error)

def normalize_weights(examples):
    "given example weights, rescales so that sum is size"
    size = len(examples)
    # find total weight by iterating through examples
    totalWeight = sum([examples[i].weight for i in xrange(size)])

    # normalize weights so all weights sum to number of examples
    for i in xrange(size):
        examples[i].weight = examples[i].weight*size/totalWeight

def update_weights(dt, examples):
    "applies math formulas to update weights based on dt"
    size = len(examples)
    total = 0.
    error = calc_error(dt, examples)

    # change weights depending on accuracy
    for i in xrange(size):
        if examples[i].attrs[9] == classify(dt, examples[i]):
            examples[i].weight = examples[i].weight*math.exp(-1.*alpha(error))
        else:
            examples[i].weight = examples[i].weight*math.exp(alpha(error))
    # normalize weights
    normalize_weights(examples)

def new_weights(dt, examples):
    "calculates new weights for examples and decision tree weight"
    error = calc_error(dt, examples)
    treeWeight = alpha(error)

    update_weights(dt, examples)
    return treeWeight

###########################################################################

def AdtAccuracy(adt, examples):
  """ returns correct/total """
  correct = 0.
  size = len(examples)

  for i in xrange(size):
    if adt.classify(examples[i]) == examples[i].attrs[-1]:
      correct += 1
  return correct/size

class adaBoostTree:
  def __init__(self, dataset):
    """ self.dts: List of decision trees
        self.weights: List of weights that correspond to dts
    """
    # Initial tree with no boosting
    dt = learn(dataset)

    self.dataset = dataset
    self.weights = [new_weights(dt, dataset.examples)]
    self.dts = [dt]

  def boost(self):
    dt = learn(self.dataset)
    weight = new_weights(dt, self.dataset.examples)
    self.dts.append(dt)
    self.weights.append(weight)

  def classify(self, example):
    """ Defaults to positive. """
    votes = [0.,0.]
    for dt, weight in zip(self.dts, self.weights):
      classification = dt.predict(example)
      votes[classification] += weight
    return 0 if votes[0] > votes[1] else 1

#---------

def main():
    arguments = validateInput(sys.argv)
    noisyFlag, pruneFlag, valSetSize, maxDepth, boostRounds = arguments
    print noisyFlag, pruneFlag, valSetSize, maxDepth, boostRounds

    # Read in the data file
    
    if noisyFlag:
        f = open("noisy.csv")
    else:
        f = open("data.csv")

    data = parse_csv(f.read(), " ")
    dataset = DataSet(data)
    
    # Copy the dataset so we have two copies of it
    examples = dataset.examples[:]
 
    dataset.examples.extend(examples)
    dataset.max_depth = maxDepth
    if boostRounds != -1:
      dataset.use_boosting = True
      dataset.num_rounds = boostRounds

    # ====================================
    # WRITE CODE FOR YOUR EXPERIMENTS HERE
    # ====================================

    # Usage: main.py [-n] [-p [valSetSize]] [-b boostingRounds] [-d weakLearnerDepth]
    # Sanity checks
    if maxDepth==0:
      sys.exit("maxDepth cannot be 0. Exiting...")
    if pruneFlag and (boostRounds > 0):
      sys.exit("Cannot prune and boost at the same time. Exiting...")
    if pruneFlag and (maxDepth > 0):
      sys.exit("Cannot prune and have maxDepth at the same time. Exiting...")

    # Maintain a reference to all the examples
    examples = dataset.examples

    # Part 2 (a)
    if not pruneFlag and (maxDepth==-1) and (boostRounds==-1):
      testPerformanceSum = 0.
      trainPerformanceSum = 0.

      # 10-fold cross validation
      for i in xrange(0, 100, 10):
        # Load the correct subset of examples; train
        dataset.examples = examples[i:i+90]
        dt = learn(dataset)

        # Record performance
        trainPerformanceSum += accuracy(dt, examples[i:i+90])
        testPerformanceSum += accuracy(dt, examples[i+90:i+100])

      print 'Average cross-validated training performance: %s' % \
        str(trainPerformanceSum / 10)
      print 'Average cross-validated test performance: %s' % \
        str(testPerformanceSum / 10)

    # Part 2 (b)
    elif pruneFlag:
      test_results = []
      train_results = []

      # For validation set range [1,80]
      for i in xrange(1,81):
          testPerformanceSum = 0.
          trainPerformanceSum = 0.

          # Take the average of 10 different validation sets
          for j in xrange(0,100,10):
              # Partition: trainStart:validationStart:testStart:end
              trainStart = j
              validationStart = j+90-i
              testStart = j+90
              end = j+100

              dataset.examples = examples[trainStart:validationStart]
              dt = learn(dataset)
              dt = prune(dt, dt, examples[validationStart:testStart])

              trainPerformanceSum += accuracy(dt, examples[trainStart:validationStart])
              testPerformanceSum += accuracy(dt, examples[testStart:end])

          train_results.append(trainPerformanceSum/10)
          test_results.append(testPerformanceSum/10)

      # Plot
      plt.clf()

      xs = range(1,81)

      plt.plot(xs, train_results, '-b')
      plt.plot(xs, test_results, '-r')
      plt.ylabel('Performance')
      plt.xlabel('Validation Set Size')
      if noisyFlag:
        plt.title('Noisy Data Pruning Results')
      else:
        plt.title('Non-Noisy Pruning Results')
      plt.legend(["training", "test"], 'best')
      plt.axis([0,80,.7,1.05])
      plt.show()

    # Part 3
    elif boostRounds > -1:
      test_results = [0.]*boostRounds
      train_results = [0.]*boostRounds

      if maxDepth > 0:
        dataset.max_depth = maxDepth

      # Take the average of 10 different training sets
      for j in xrange(0,100,10):
        # Partition: trainStart:testStart:end
        trainStart = j
        testStart = j+90
        end = j+100

        # Clear weights
        for e in examples:
          e.weight = 1

        dataset.examples = examples[trainStart:testStart]
        adt = adaBoostTree(dataset)

        # Perform boosting
        for i in xrange(boostRounds):
          adt.boost()
          test_results[i] += AdtAccuracy(adt, examples[testStart:end])
          train_results[i] += AdtAccuracy(adt, examples[trainStart:testStart])

      test_results = [e/10. for e in test_results]
      train_results = [e/10. for e in train_results]

      # Plot
      plt.clf()

      xs = range(1, boostRounds+1)

      plt.plot(xs, test_results, '-r')
      plt.plot(xs, train_results, '-b')
      plt.ylabel('Performance')
      plt.xlabel('Number of Boosting Rounds')
      if noisyFlag:
        plt.title('AdaBoost Noisy Depth 1 Results')
      else:
        plt.title('AdaBoost Non-Noisy Depth 1 Results')
      plt.legend(["test", "training"], 'best')
      plt.axis([0,30,.75,1])
      plt.show()

main()


    
