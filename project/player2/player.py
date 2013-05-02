import common
from ann.ann import *
from ann.ann_impl import *
import game_interface as game

learning_rate = 0.5

network = None
lastScore = None
lastImage = None

# ANN training
def get_move(view):
    global network, lastScore, lastImage
    # Initialize network
    if network == None:
        network = HiddenNetwork(15)
    network.InitializeWeights()

    # Train last round
    if lastImage != None:
        target = [1.0,0.0] if view.GetLife() > lastScore else [0.0,1.0]
        Backprop(network.network, lastImage, target, learning_rate)

    lastImage = [float(x) for x in view.GetImage()]
    # Train this round
    if view.GetPlantInfo() == game.STATUS_NUTRITIOUS_PLANT or \
            view.GetPlantInfo == game.STATUS_POISONOUS_PLANT:
        target = [1.0,0.0] if view.GetPlantInfo() == game.STATUS_NUTRITIOUS_PLANT \
            else [0.0,1.0]

        Backprop(network.network, lastImage, target, learning_rate)
        lastImage = None
    # Train next round
    else:
        lastScore = view.GetLife()

    direction = random.choice([game.UP,game.DOWN,game.LEFT,game.RIGHT])
    return (direction, True)
