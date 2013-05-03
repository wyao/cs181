import common
import time
import random
import game_interface as game
import py.test

# "MACROS"
LEARNING_RATE = 0.1
GAMMA = 0.5
[PROB,ENERGY,VISITED,STEPS,STATUS] = range(5)
[LOW,MID,HIGH] = range(3)
SCORE_MULT = 3
K_MEMORY = 2

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

Q = {}
START_SCORE = None
last_action = None
last_state = None

def get_move(view, options):
    global Q, START_SCORE, states

    # Setup
    if Q == {}:
        START_SCORE = view.GetLife()

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

    if view.GetPlantInfo() == game.STATUS_NUTRITIOUS_PLANT:
        eat = True
    elif view.GetPlantInfo() == game.STATUS_UNKNOWN_PLANT:
        eat = random.choice([True,False])
    else:
        eat = False
    direction = random.choice(moves)
    # time.sleep(0.1)
    return (direction, eat)

def round_down(n, d):
    return n - (n % d)

def reward(s):
    return s[ENERGY]

def Q_learning(s, s_, a):
    """ s is old score, s_ is new (current) score """
    global Q
    x = GAMMA * max([Q[s_][a_] for a_ in actions])
    y = reward(s) + x - Q[s][a]
    Q[s][a] += LEARNING_RATE * y