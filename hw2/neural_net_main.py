from data_reader import *
from neural_net import *
from neural_net_impl import *
import sys
import random
import pickle


def parseArgs(args):
  """Parses arguments vector, looking for switches of the form -key {optional value}.
  For example:
    parseArgs([ 'main.py', '-e', 20, '-r', 0.1, '-m', 'Simple' ]) = { '-e':20, '-r':5, '-t': 'simple' }"""
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
  assert '-e' in args_map, "A number of epochs should be specified with the flag -e (ex: -e 10)"
  assert '-r' in args_map, "A learning rate should be specified with the flag -r (ex: -r 0.1)"
  assert '-t' in args_map, "A network type should be provided. Options are: simple | hidden | custom"
  return(args_map)

def main():

  # Parsing command line arguments
  args_map = validateInput(sys.argv)
  epochs = int(args_map['-e'])
  rate = float(args_map['-r'])
  networkType = args_map['-t']
  hidden = int(args_map['-h']) if '-h' in args_map else 15
  prob = float(args_map['-p']) if '-p' in args_map else 0.8

  # Load in the training data.

  tests = DataReader.GetImages('test-1k.txt', -1)
  if '-d' not in args_map:
    images = DataReader.GetImages('training-9k.txt', -1)
  else:
    images = DataReader.GetImages('test-1k.txt', -1)
  for image in images:
    assert len(image.pixels) == 14
    assert len(image.pixels[0]) == 14

  # Load the validation set.
  validation = DataReader.GetImages('validation-1k.txt', -1)
  for image in validation:
    assert len(image.pixels) == 14
    assert len(image.pixels[0]) == 14

  # Initializing network

  if networkType == 'simple':
    network = SimpleNetwork()
  if networkType == 'hidden':
    network = HiddenNetwork(hidden)
  if networkType == 'custom':
    network = CustomNetwork(hidden, prob)

  # Hooks user-implemented functions to network
  network.FeedForwardFn = FeedForward
  network.TrainFn = Train

  # Initialize network weights
  network.InitializeWeights()
  

  # Displays information
  print '* * * * * * * * *'
  print 'Parameters => Epochs: %d, Learning Rate: %f' % (epochs, rate)
  print 'Type of network used: %s' % network.__class__.__name__
  print ('Input Nodes: %d, Hidden Nodes: %d, Output Nodes: %d' %
         (len(network.network.inputs), len(network.network.hidden_nodes),
          len(network.network.outputs)))
  print '* * * * * * * * *'
  # Train the network.
  log = network.Train(images, validation, tests, rate, epochs)
  f = open(''.join(['log/']+['%s-%s_' % (k,v) for k,v in args_map.items()] + ['.log']), 'w')
  pickle.dump(log, f)
  f.close()

if __name__ == "__main__":
  main()
