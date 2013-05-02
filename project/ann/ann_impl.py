from ann import NeuralNetwork, NetworkFramework, Node, Target, Input
import random
import math
import py.test

INPUT = 1
HIDDEN = 2
OUTPUT = 3

def FeedForward(network, input):
  """
  This function propagates the inputs through the network. That is,
  it modifies the *raw_value* and *transformed_value* attributes of the
  nodes in the network, starting from the input nodes.
  """
  network.CheckComplete()

  # Propagation helper; takes hidden and output nodes
  def propagate(nodes):
    for n in nodes:
      # Compute raw value
      n.raw_value = network.ComputeRawValue(n)
      # Compute transformed value
      n.transformed_value = network.Sigmoid(n.raw_value)

  # Assign input values to input nodes
  for i in xrange(len(input)):
    network.inputs[i].raw_value = input[i]
    network.inputs[i].transformed_value = input[i]

  propagate(network.hidden_nodes + network.outputs)

def Backprop(network, input, target, learning_rate):
  network.CheckComplete()

  # First propagate the input through the network
  FeedForward(network, input)

  # Calculate output errors and deltas
  for (y,out_node) in zip(target, network.outputs):
    out_node.error = y - out_node.transformed_value
    out_node.delta = out_node.error * network.SigmoidPrime(out_node.raw_value)

  # Backpropagate in reverse topoligical order
  nodes = network.inputs + network.hidden_nodes
  nodes.reverse()

  for node in nodes:
    node.error = 0.
    for child,w in zip(node.forward_neighbors, node.forward_weights):
      if node.inputs:
        node.error += w.value * child.delta
      w.value += learning_rate * node.transformed_value * child.delta
    if node.inputs:
      node.delta = node.error * network.SigmoidPrime(node.raw_value)
      node.fixed_weight.value += learning_rate * node.delta

def Train(network, inputs, targets, learning_rate, epochs):
  network.CheckComplete()
  for _ in xrange(epochs):
    for input,target in zip(inputs, targets):
      Backprop(network, input, target, learning_rate)

class EncodedNetworkFramework(NetworkFramework):
  def __init__(self):
    """
    Initializatio.
    YOU DO NOT NEED TO MODIFY THIS __init__ method
    """
    super(EncodedNetworkFramework, self).__init__() # < Don't remove this line >
    
  # <--- Fill in the methods below --->

  def EncodeLabel(self, label):
    """
    Computes the distributed encoding of a given label.
    """
    target = Target()
    target.values = [1.0 if i == label else 0.0 for i in xrange(10)]
    return target

  def GetNetworkLabel(self):
    """
    The function looks for the transformed_value of each output, then decides 
    which label to attribute to this list of outputs. The idea is to 'line up'
    the outputs, and consider that the label is the index of the output with the
    highest *transformed_value* attribute
    """
    transformed_values = map(lambda n: n.transformed_value, self.network.outputs)
    return transformed_values.index(max(transformed_values))

  def Convert(self, image):
    """
    The *image* arguments has 2 attributes: *label* which indicates
    the digit represented by the image, and *pixels* a matrix 14 x 14
    represented by a list (first list is the first row, second list the
    second row, ... ), containing numbers whose values are comprised
    between 0 and 256.0. The function transforms this into a unique list
    of 14 x 14 items, with normalized values (that is, the maximum possible
    value should be 1).
    """
    input = Input()
    input.values = [item/256. for sublist in image.pixels for item in sublist]
    return input

  def InitializeWeights(self):
    """
    Initializes with offline weights or with random values
    between [-0.01, 0.01].
    """
    for weight in self.network.weights:
      weight.value = random.uniform(-.01,.01)

  def ExportWeights(self):
    pass

class HiddenNetwork(EncodedNetworkFramework):
  def __init__(self, number_of_hidden_nodes=15):
    super(HiddenNetwork, self).__init__()

    # 1) Adds an input node for each pixel
    network = self.network
    for _ in xrange(36):
      network.AddNode(Node(), INPUT)

    # 2) Adds the hidden layer
    for _ in xrange(number_of_hidden_nodes):
      node = Node()
      for input_node in network.inputs:
        node.AddInput(input_node, None, network)
      network.AddNode(node, HIDDEN)

    # 3) Adds an output node for nutritious and poisonous.
    for _ in xrange(2):
      node = Node()
      for hidden_node in network.hidden_nodes:
        node.AddInput(hidden_node, None, network)
      network.AddNode(node, OUTPUT)
