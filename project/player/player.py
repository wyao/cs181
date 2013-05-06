import random
import math
import cPickle
import game_interface as game
from ann import *
from ann_impl import *
import py.test

# Constants
LEARNING_RATE = 0.1
GAMMA = 0.5
[PROB,ENERGY,VISITED,DISTANCE,STATUS,INCREASED] = range(6)
[LOW,MID,HIGH] = range(3)
[DIRECTION,EAT] = range(2)
MAX_ENERGY = 200
SCORE_INCR = 10
MAX_DISTANCE = 100
# Dynamically generated constants
START_SCORE = None
moves = [game.UP,game.DOWN,game.LEFT,game.RIGHT]
actions = [(m, True) for m in moves] + [(m, False) for m in moves]
states = []
status_ = [game.STATUS_UNKNOWN_PLANT, game.STATUS_NUTRITIOUS_PLANT, \
    game.STATUS_POISONOUS_PLANT, game.STATUS_NO_PLANT]
# Globals
Q = {}
visited_locations = {}
network = None
last_score = None

def get_move(view):
    global states, Q, START_SCORE, network, last_score
    score = view.GetLife()
    info = view.GetPlantInfo()
    loc = (view.GetXPos(), view.GetYPos())
    # Setup
    if Q == {}:
        START_SCORE = score
        last_score = score + 1
        # Initialize states
        for p in [LOW,MID,HIGH]:
            for e in range(0, MAX_ENERGY + 1, 10):
                for v in [True,False]:
                    for d in xrange(MAX_DISTANCE + 1):
                        for s in status_:
                            for i in [True, False]:
                                states.append((p,e,v,d,s,i))
        # Initialize Q
        f = open("player/20000.q", "r")
        Q = cPickle.load(f)
        f.close()
        # Setup neural network
        if network == None:
            network = SimpleNetwork()
            network.FeedForwardFn = FeedForward
            network.TrainFn = Train
            network.InitializeWeights("player/e5_l03.weights")
    # Construct current state
    state = (get_prob(view), round_down(score), visited(loc), \
        distance(loc), info, score > last_score)
    # Select action based on learned Q values
    action = exploit(state)
    # Update
    last_score = score
    return (action[DIRECTION], action[EAT])

def exploit(state):
    choices = [(value,a) for (a,value) in Q[state].iteritems()]
    # If first time seeing this state, bias using our belief of plant type
    if max(choices)[0] == 0:
        if state[INCREASED]:
            return (random.choice(moves), True)
        if state[STATUS] == game.STATUS_NUTRITIOUS_PLANT:
            return (random.choice(moves), True)
        if state[PROB] == HIGH:
            return (random.choice(moves), True)
        if state[PROB] == MID:
            return (random.choice(moves), random.choice([True,False]))
        return (random.choice(moves), False)
    # Else exploit using Q
    return max(choices)[1]

def round_down(n):
    return min(n - (n % SCORE_INCR), MAX_ENERGY)

def get_prob(view):
    if network.Classify(view.GetImage()) == 1:
        return HIGH
    return LOW

def distance(loc):
    x,y = loc
    x,y = float(x),float(y)
    return min(int(math.sqrt(math.pow(x,2) + math.pow(y,2))), MAX_DISTANCE)


# Returns whether location has been visited and stores visits
def visited(loc):
    if loc in visited_locations:
        return True
    visited_locations[loc] = True
    return False
