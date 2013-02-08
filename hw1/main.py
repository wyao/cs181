# main.py
# -------
# Irene Chen and Willie Yao

from dtree import *
import sys
import py.test
import matplotlib.pyplot as plt
from pylab import *

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
    learner.train( dataset)
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
  """ Returns majority classfication, or None if pruning not necessary.
      Defaults to positive.
  """
  labelCount = [0,0]
  for attr in dt.branches:
    if dt.branches[attr].nodetype is NODE:
      return None
    labelCount[dt.branches[attr].classification] += 1
  return 0 if labelCount[0] < labelCount[1] else 1


def prune(dt, root, examples):
  """ Assumptions:
        Pruning terminates as soon as a node rejects pruning.
        Labels must be binary.
        This implementation cannot prune a tree that has a depth < 2.
  """
  for attr in dt.branches:
    # Ignore leaves
    if dt.branches[attr].nodetype is NODE:
      # Bottom-up pruning
      dt.branches[attr] = prune(dt.branches[attr], root, examples)

      # Prune (if applicable)
      classfication = getClassification(dt.branches[attr])
      if classfication is not None:
        # Test performance without pruning
        originalPerformance = accuracy(root, examples)
        originalBranch = dt.branches[attr]

        # Test performance with pruning
        dt.branches[attr] = DecisionTree(LEAF, classification=classfication)
        if accuracy(root, examples) < originalPerformance:
          dt.branches[attr] = originalBranch
  return dt

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

    examples = dataset.examples

    # Part 2 (a)
    if not pruneFlag:
      testPerformanceSum = 0.
      trainPerformanceSum = 0.

      # 10-fold cross validation
      for i in xrange(0, 100, 10):
        # Load the correct subset of examples
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
              end = j + 100

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
      plt.ylabel('y')
      plt.xlabel('x')
      plt.title('Train Results')
      plt.legend(["training", "test"], 'best')
      plt.show()

main()


    
