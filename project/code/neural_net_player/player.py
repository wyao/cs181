import common
from code.ann import *
from code.ann_impl import *
import game_interface as game

learning_rate = 0.5

network = None
lastScore = None
lastImage = None
correct = 0.
instances = 0.

firstMove = True
plant_data = []
lastImages = []

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

def new_get_move(view, options):
    global firstMove, lastScore, lastImages, plant_data

    if firstMove and lastImages == []:
        InitalizePlants(options.out_file)
        firstMove = False

    # Train last round
    if lastImages != []:
        if options.train == 3:
            # gained points -> nutritious plant -> 1
            plantStatus = 1 if view.GetLife() > lastScore else 0
            for i in lastImages:
                classification = (plantStatus, i)
                plant_data.append(classification)

    # Bookkeeping
    plantInfo = view.GetPlantInfo()
    if plantInfo == game.STATUS_NO_PLANT:
        lastImages = []
    else:
        lastImages.append([float(x) for x in view.GetImage()])
        lastScore = view.GetLife()

    direction = random.choice([game.UP,game.DOWN,game.LEFT,game.RIGHT])
    return (direction, True)

def ExportPlants(file_name):
    # weights = [w.value for w in self.network.weights]
    f = open(file_name,"w")
    pickle.dump(plant_data, f)
    f.close()

def InitalizePlants(file_name):
    """
    Initializes with offline weights or with random values
    between [-0.01, 0.01].
    """
    global plant_data

    try:
      f = open(file_name)
      plant_data = pickle.load(f)
      f.close()
    except:
        pass