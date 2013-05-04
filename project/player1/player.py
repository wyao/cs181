import common
import time
import random
import game_interface as game
import py.test

# Constants
LEARNING_RATE = 0.1
GAMMA = 0.5
[PROB,ENERGY,VISITED,STEPS,STATUS] = range(5)
[LOW,MID,HIGH] = range(3)
[DIRECTION,EAT] = range(2)
SCORE_MULT = 3
SCORE_INCR = 10
K_MEMORY = 2
# Dynamically generated constants
START_SCORE = None
moves = [game.UP,game.DOWN,game.LEFT,game.RIGHT]
actions = [(m, True) for m in moves] + [(m, False) for m in moves]
states = []
steps_ = [()]
for _ in xrange(K_MEMORY):
    new_steps = []
    for l in steps_:
        for m in moves+[None]:
            x = list(l)
            x.append(m)
            new_steps.append(tuple(x))
    steps_ = new_steps
status_ = [game.STATUS_UNKNOWN_PLANT, game.STATUS_NUTRITIOUS_PLANT, \
    game.STATUS_POISONOUS_PLANT, game.STATUS_NO_PLANT]
# Globals
Q = {}
last_action = None
last_state = None
last_location = None
visited_locations = {}

def get_move(view, options):
    global states, Q, START_SCORE, last_action, last_state, last_location
    score = view.GetLife()
    info = view.GetPlantInfo()
    loc = (view.GetXPos(), view.GetYPos())
    # Setup
    if Q == {}:
        START_SCORE = score
        # Initialize states
        for p in [LOW,MID,HIGH]:
            for e in range(0, SCORE_MULT * START_SCORE, 10):
                for v in [True,False]:
                    for k in steps_:
                        for s in status_:
                            states.append((p,e,v,k,s))
        # Initialize Q
        for s in states:
            Q[s] = {}
            for a in actions:
                Q[s][a] = 0.
        # Initial action
        visited(loc)
        last_action = (random.choice(moves), False)
        last_state = (0., round_down(START_SCORE), False, \
            tuple(K_MEMORY*[None]), info)
    # Perform learning / make decision
    else:
        # Construct current state
        new_steps = list(last_state[STEPS])
        del new_steps[0]
        new_steps.append(last_step(loc))
        new_steps = tuple(new_steps)
        state = (get_prob(view), round_down(score), visited(loc), new_steps, info)
        # Q-Learning
        Q_learning(last_state, state, last_action)
        # Explore
        if to_explore():
            last_action = random.choice(actions)
        # Exploit
        else:
            last_action = exploit(state)
        last_state = state
    last_location = loc
    # time.sleep(0.1)
    return (last_action[DIRECTION], last_action[EAT])

# TODO: Explore every action possible at a given state first across games
def to_explore():
    return int(random.random() < .1)

def exploit(state):
    choices = [(value,a) for (a,value) in Q[state].iteritems()]
    # If first time seeing this state, bias using our belief of plant type
    if max(choices)[0] == 0:
        if state[PROB] == HIGH:
            return (random.choice(moves), True)
        if state[PROB] == MID:
            return (random.choice(moves), random.choice([True,False]))
        return (random.choice(moves), False)
    # Else exploit using Q
    return max(choices)[1]

def round_down(n):
    return min(n - (n % SCORE_INCR), SCORE_MULT * START_SCORE)

def reward(s):
    return s[ENERGY]

def get_prob(view):
    return random.choice([LOW,MID,HIGH])

# Get the actual last step that was taken
def last_step(loc):
    global last_location
    x,y = last_location
    x_,y_ = loc
    if y_ > y:
        return game.UP
    if y_ < y:
        return game.DOWN
    if x_ < x:
        return game.LEFT
    return game.RIGHT

# Returns whether location has been visited and stores visits
def visited(loc):
    if loc in visited_locations:
        return True
    visited_locations[loc] = True
    return False

def Q_learning(s, s_, a):
    """ s is old score, s_ is new (current) score """
    global Q
    x = GAMMA * max([Q[s_][a_] for a_ in actions])
    y = reward(s) + x - Q[s][a]
    Q[s][a] += LEARNING_RATE * y
