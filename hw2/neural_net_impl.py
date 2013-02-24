from neural_net import NeuralNetwork, NetworkFramework
from neural_net import Node, Target, Input
import random
import math
import py.test

INPUT = 1
HIDDEN = 2
OUTPUT = 3

# <--- Problem 3, Question 1 --->

def FeedForward(network, input):
  """
  Arguments:
  ---------
  network : a NeuralNetwork instance
  input   : an Input instance

  Returns:
  --------
  Nothing

  Description:
  -----------
  This function propagates the inputs through the network. That is,
  it modifies the *raw_value* and *transformed_value* attributes of the
  nodes in the network, starting from the input nodes.

  Notes:
  -----
  The *input* arguments is an instance of Input, and contains just one
  attribute, *values*, which is a list of pixel values. The list is the
  same length as the number of input nodes in the network.

  i.e: len(input.values) == len(network.inputs)

  This is a distributed input encoding (see lecture notes 7 for more
  informations on encoding)

  In particular, you should initialize the input nodes using these input
  values:

  network.inputs[i].raw_value = input[i] # This is wrong
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
  for i in xrange(len(input.values)):
    network.inputs[i].raw_value = input.values[i]
    network.inputs[i].transformed_value = input.values[i]

  propagate(network.hidden_nodes + network.outputs)

#< --- Problem 3, Question 2

def Backprop(network, input, target, learning_rate):
  """
  Arguments:
  ---------
  network       : a NeuralNetwork instance
  input         : an Input instance
  target        : a target instance
  learning_rate : the learning rate (a float)

  Returns:
  -------
  Nothing

  Description:
  -----------
  The function first propagates the inputs through the network
  using the Feedforward function, then backtracks and update the
  weights.

  Notes:
  ------
  The remarks made for *FeedForward* hold here too.

  The *target* argument is an instance of the class *Target* and
  has one attribute, *values*, which has the same length as the
  number of output nodes in the network.

  i.e: len(target.values) == len(network.outputs)

  In the distributed output encoding scenario, the target.values
  list has 10 elements.

  When computing the error of the output node, you should consider
  that for each output node, the target (that is, the true output)
  is target[i], and the predicted output is network.outputs[i].transformed_value.
  In particular, the error should be a function of:

  target[i] - network.outputs[i].transformed_value
  
  """
  network.CheckComplete()

  # First propagate the input through the network
  FeedForward(network, input)

  # Calculate output errors and deltas
  for (y,out_node) in zip(target, network.outputs):
    out_node.error = y - out_node.transformed_value
    out_node.delta = out_node.error * out_node.transformed_value * \
      (1. - out_node.transformed_value)

  def back_prop(nodes, learning_rate):
    """ Calculate the error, delta, and forward_weights of given node.
        Takes hidden and input nodes.
        Can be made even more efficient by splitting into 2 functions.
    """
    for node in nodes:
      # Backprop
      for i in xrange(len(node.forward_neighbors)):
        child, w = node.forward_neighbors[i], node.forward_weights[i]
        # Compute error if not input node
        node.error = 0.
        if node.inputs:
          node.error += w.value * child.delta
        # Update forward_weights
        node.forward_weights[i].value += \
          learning_rate * node.transformed_value * child.delta
      # Compute delta if not input node
      if node.inputs:
        node.delta = node.error * network.SigmoidPrime(node.transformed_value) #node.transformed_value * (1. - node.transformed_value)

  # Backpropagate; note the reverse topoligical order
  back_prop(network.hidden_nodes[::-1] + network.inputs, learning_rate)

# <--- Problem 3, Question 3 --->

def Train(network, inputs, targets, learning_rate, epochs):
  """
  Arguments:
  ---------
  network       : a NeuralNetwork instance
  inputs        : a list of Input instances
  targets       : a list of Target instances
  learning_rate : a learning_rate (a float)
  epochs        : a number of epochs (an integer)

  Returns:
  -------
  Nothing

  Description:
  -----------
  This function should train the network for a given number of epochs. That is,
  run the *Backprop* over the training set *epochs*-times
  """
  network.CheckComplete()
  for _ in xrange(epochs):
    for input,target in zip(inputs, targets):
      Backprop(network, input, target, learning_rate)

# <--- Problem 3, Question 4 --->

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
    Arguments:
    ---------
    label: a number between 0 and 9

    Returns:
    ---------
    a list of length 10 representing the distributed
    encoding of the output.

    Description:
    -----------
    Computes the distributed encoding of a given label.

    Example:
    -------
    0 => [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    3 => [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    Notes:
    ----
    Make sure that the elements of the encoding are floats.
    
    """
    return [1.0 if i == label else 0.0 for i in xrange(10)]

  def GetNetworkLabel(self):
    """
    Arguments:
    ---------
    Nothing

    Returns:
    -------
    the 'best matching' label corresponding to the current output encoding

    Description:
    -----------
    The function looks for the transformed_value of each output, then decides 
    which label to attribute to this list of outputs. The idea is to 'line up'
    the outputs, and consider that the label is the index of the output with the
    highest *transformed_value* attribute

    Example:
    -------

    # Imagine that we have:
    map(lambda node: node.transformed_value, self.network.outputs) => [0.2, 0.1, 0.01, 0.7, 0.23, 0.31, 0, 0, 0, 0.1, 0]

    # Then the returned value (i.e, the label) should be the index of the item 0.7,
    # which is 3
    
    """
    transformed_values = map(lambda n: n.transformed_value, self.network.outputs)
    return transformed_values.index(max(transformed_values))

  def Convert(self, image):
    """
    Arguments:
    ---------
    image: an Image instance

    Returns:
    -------
    an instance of Input

    Description:
    -----------
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
    Arguments:
    ---------
    Nothing

    Returns:
    -------
    Nothing

    Description:
    -----------
    Initializes the weights with random values between [-0.01, 0.01].

    Hint:
    -----
    Consider the *random* module. You may use the the *weights* attribute
    of self.network.
    
    """
    for weight in self.network.weights:
      weight.value = random.uniform(-.01,.01)

#<--- Problem 3, Question 6 --->

class SimpleNetwork(EncodedNetworkFramework):
  def __init__(self):
    """
    Arguments:
    ---------
    Nothing

    Returns:
    -------
    Nothing

    Description:
    -----------
    Initializes a simple network, with 196 input nodes,
    10 output nodes, and NO hidden nodes. Each input node
    should be connected to every output node.
    """
    super(SimpleNetwork, self).__init__() # < Don't remove this line >

    # 1) Adds an input node for each pixel.    
    network = self.network
    for _ in xrange(196): # 196 = 14*14
      node = Node()
      network.AddNode(node, INPUT)

    # 2) Add an output node for each possible digit label.
    for _ in xrange(10):
      node = Node()
      for input_node in network.inputs:
        node.AddInput(input_node, None, network)
      network.AddNode(node, OUTPUT)

#<---- Problem 3, Question 7 --->

class HiddenNetwork(EncodedNetworkFramework):
  def __init__(self, number_of_hidden_nodes=15):
    """
    Arguments:
    ---------
    number_of_hidden_nodes : the number of hidden nodes to create (an integer)

    Returns:
    -------
    Nothing

    Description:
    -----------
    Initializes a network with a hidden layer. The network
    should have 196 input nodes, the specified number of
    hidden nodes, and 10 output nodes. The network should be,
    again, fully connected. That is, each input node is connected
    to every hidden node, and each hidden_node is connected to
    every output node.
    """
    super(HiddenNetwork, self).__init__() # < Don't remove this line >

    # 1) Adds an input node for each pixel
    network = self.network
    for _ in xrange(196): # 196 = 14*14
      node = Node()
      network.AddNode(node, INPUT)

    # 2) Adds the hidden layer
    for _ in xrange(30):
      node = Node()
      for input_node in network.inputs:
        node.AddInput(input_node, None, network)
      network.AddNode(node, HIDDEN)

    # 3) Adds an output node for each possible digit label.
    for _ in xrange(10):
      node = Node()
      for hidden_node in network.hidden_nodes:
        node.AddInput(hidden_node, None, network)
      network.AddNode(node, OUTPUT)

#<--- Problem 3, Question 8 ---> 

class CustomNetwork(EncodedNetworkFramework):
  def __init__(self):
    """
    Arguments:
    ---------
    Your pick.

    Returns:
    --------
    Your pick

    Description:
    -----------
    Surprise me!
    """
    super(CustomNetwork, self).__init__() # <Don't remove this line>
    pass
