#
# Darts playing model for CS181.
#

import sys
import time
import random
import throw
import mdp
import modelbased
import modelfree

GAMMA = .5
EPOCH_SIZE = 10


# <CODE HERE>: Complete this function, which should return a
# list of all possible states.
def get_states():
  # should return a **list** of states. Each state should be an integer.
  return range(0, throw.START_SCORE+1)

# Returns a list of all possible actions, or targets, which include both a
# wedge number and a ring.
def get_actions():

  actions = []
  
  for wedge in throw.wedges:
    actions = actions + [throw.location(throw.CENTER, wedge)]
    actions = actions + [throw.location(throw.INNER_RING, wedge)]
    actions = actions + [throw.location(throw.FIRST_PATCH, wedge)]
    actions = actions + [throw.location(throw.MIDDLE_RING, wedge)]
    actions = actions + [throw.location(throw.SECOND_PATCH, wedge)]
    actions = actions + [throw.location(throw.OUTER_RING, wedge)]
    
  return actions

# <CODE HERE>: Define the reward function
def R(s,a):
  # takes a state s and action a
  # returns the reward for completing action a in state s
  """ Let reward be the expected score gained given the action
  """
  reward = 0
  for score in xrange(1, s+1):
    reward += mdp.T(a, s, s - score) * score
  return reward
  # if s == 0:
  #   return 1.
  # return 0.

# Play a single game 
def play(method, GAMMA, d=None):
    score = throw.START_SCORE
    turns = 0
    
    if method == "mdp":
        target = mdp.start_game(GAMMA)
    else:
        target = modelfree.start_game()
        
    targets = []
    results = []
    while(True):
        turns = turns + 1
        result = throw.throw(target)
        targets.append(target)
        results.append(result)
        raw_score = throw.location_to_score(result)
        if d:
            if d[score] == None:
                d[score] = throw.location_to_score(target)
            else:
                assert(d[score] == throw.location_to_score(target))
        # print "Target: wedge", target.wedge,", ring", target.ring
        # print "Result: wedge", result.wedge,", ring", result.ring
        # print "Raw Score:", raw_score
        # print "Score:", score
        if raw_score <= score:
            score = int(score - raw_score)
        # else:
        #     print
        #     print "TOO HIGH!"
        if score == 0:
            break

        if method == "mdp":
            target = mdp.get_target(score)
        else:
            target = modelfree.get_target(score)
    # print "WOOHOO!  It only took", turns, " turns"
    #end_game(turns)
    return turns

# Play n games and return the average score. 
def test(n, method, GAMMA):
    score = 0
    if n > 0:
        for i in range(n):
            score += play(method, GAMMA)

        print "Average turns = ", float(score)/float(n)
    else:
        d = {}
        for i in xrange(1,throw.START_SCORE+1):
            d[i] = None
        def determined_policy(d):
            for v in d.values():
                if v == None:
                    return False
            return True

        while not determined_policy(d):
            play(method, GAMMA, d)

        print d
    return score

# <CODE HERE>: Feel free to modify the main function to set up your experiments.
def main():
    throw.init_board()
    num_games = 1000

#************************************************#
# Uncomment the lines below to run the mdp code, #
# using the simple dart thrower that matches     #
# the thrower specified in question 2.           #
#*************************************************

# Default is to solve MDP and play 1 game
    throw.use_simple_thrower()
    GAMMA = 0.
    for i in range(0, 11):
        print GAMMA,
        test(-1, "mdp", GAMMA)
        GAMMA +=.1

#*************************************************#
# Uncomment the lines below to run the modelbased #
# code using the complex dart thrower.            #
#*************************************************#

# Seed the random number generator -- the default is
# the current system time. Enter a specific number
# into seed() to keep the dart thrower constant across
# multiple calls to main().
# Then, initialize the throwing model and run
# the modelbased algorithm.
    #random.seed()
    #throw.init_thrower()
    #modelbased.modelbased(GAMMA, EPOCH_SIZE, num_games)

#*************************************************#
# Uncomment the lines below to run the modelfree  #
# code using the complex dart thrower.            #
#*************************************************#

# Plays 1 game using a default player. No modelfree
# code is provided. 
    #random.seed()
    #throw.init_thrower()
    #test(1, "modelfree")


if __name__ =="__main__":
    main()




