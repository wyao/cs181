from random import *
import throw
import darts
import py.test
 
# The default player aims for the maximum score, unless the
# current score is less than the number of wedges, in which
# case it aims for the exact score it needs. 
#  
# You may use the following functions as a basis for 
# implementing the Q learning algorithm or define your own 
# functions.

LEARNING_RATE = 0.1

Q = {}

actions = None
s_old = None
a_old = None
num_iterations = 0

def start_game():
    global actions, s_old, a_old
    if actions == None:
        print "GAMMA: ", darts.GAMMA
        print "LEARNING_RATE: ", LEARNING_RATE
        print "strategy: ", darts.strategy
        actions = darts.get_actions()
        for s in darts.get_states():
            Q[s] = {}
            for a in actions:
                Q[s][a] = 0.
    s_old = throw.START_SCORE
    a_old = actions[-15]
    return a_old

def get_target(s_):
    global s_old, a_old, num_iterations
    Q_learning(s_old, s_, a_old)

    to_explore = 0
    if darts.strategy == 1:
        to_explore = ex_strategy_one()
    else:
        num_iterations += 1
        to_explore = ex_strategy_two()

    if to_explore:
        a_old = choice(actions)
    else:
        choices = [(value,a) for (a,value) in Q[s_].iteritems()]
        """ If first time at state, shoot for 24 (the max) if score >= max
            Else pick random action that does not exceed score
        """
        if max(choices)[0] == 0:
            if s_ < 24:
                a_old = choice(actions)
                while (throw.location_to_score(a_old) > s_):
                    a_old = choice(actions)
            else:
                a_old = actions[-15]
            return a_old
        # Else pick action with max Q
        a_old = max(choices)[1]
    s_old = s_
    return a_old


# Exploration/exploitation strategy one.
def ex_strategy_one():
  return int(random() < .1)


# Exploration/exploitation strategy two.
def ex_strategy_two():
  prob = max((1. / (float(num_iterations)+10.)), 0.005)
  return int(random() < prob)


# The Q-learning algorithm:
def Q_learning(s, s_, a):
    """ s is old score, s_ is new (current) score """
    x = darts.GAMMA * max([Q[s_][a_] for a_ in actions])
    y = darts.R(s,a) + x - Q[s][a]
    Q[s][a] += LEARNING_RATE * y
