import common
from ann.ann import *
from ann.ann_impl import *
import game_interface as game

learning_rate = 0.5

network = None
lastScore = None
lastImage = None
correct = 0.
instances = 0.

# ANN training
# Label 0: Poisonous, Label 1: Nutritious
def get_move(view, options):
    global network, lastScore, lastImage, correct, instances
    # Initialize network
    if network == None:
        network = HiddenNetwork(options.hidden)
        network.FeedForwardFn = FeedForward
        network.TrainFn = Train
        network.InitializeWeights(options.in_file)

    # Train last round
    if lastImage != None:
        if options.train == 1:
            target = [0.0,1.0] if view.GetLife() > lastScore else [1.0,0.0]
            Backprop(network.network, lastImage, target, learning_rate)
        else:
            if network.Classify(lastImage) == int(view.GetLife() > lastScore):
                correct += 1
            instances += 1

    # Bookkeeping
    plantInfo = view.GetPlantInfo()
    if plantInfo == game.STATUS_NO_PLANT:
        lastImage = None
    else:
        lastImage = [float(x) for x in view.GetImage()]
        lastScore = view.GetLife()

    # Train this round
    if plantInfo == game.STATUS_NUTRITIOUS_PLANT or \
            plantInfo == game.STATUS_POISONOUS_PLANT:
        if options.train == 1:
            target = [0.0,1.0] if plantInfo == game.STATUS_NUTRITIOUS_PLANT \
                else [0.0,0.0]
            Backprop(network.network, lastImage, target, learning_rate)
        else:
            if network.Classify(lastImage) == \
                    int(plantInfo == game.STATUS_NUTRITIOUS_PLANT):
                correct += 1
            instances += 1

        lastImage = None

    direction = random.choice([game.UP,game.DOWN,game.LEFT,game.RIGHT])
    return (direction, True)
