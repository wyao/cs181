import common
from ann.ann import *
from ann.ann_impl import *
import game_interface as game

learning_rate = 0.5

network = None
lastScore = None
lastImage = None

# ANN training
# Label 0: Poisonous, Label 1: Nutritious
def get_move(view):
    global network, lastScore, lastImage
    # Initialize network
    if network == None:
        network = HiddenNetwork(15)
        network.FeedForwardFn = FeedForward
        network.TrainFn = Train
        network.InitializeWeights()

    # Train last round
    if lastImage != None:
        target = [0.0,1.0] if view.GetLife() > lastScore else [1.0,0.0]
        Backprop(network.network, lastImage, target, learning_rate)

    plantInfo = view.GetPlantInfo()
    if plantInfo == game.STATUS_NO_PLANT:
        lastImage = None
    else:
        lastImage = [float(x) for x in view.GetImage()]
        lastScore = view.GetLife()

    # Train this round
    if plantInfo == game.STATUS_NUTRITIOUS_PLANT or \
            plantInfo == game.STATUS_POISONOUS_PLANT:
        target = [0.0,1.0] if plantInfo == game.STATUS_NUTRITIOUS_PLANT \
            else [0.0,0.0]
        Backprop(network.network, lastImage, target, learning_rate)
        lastImage = None

    direction = random.choice([game.UP,game.DOWN,game.LEFT,game.RIGHT])
    return (direction, True)
