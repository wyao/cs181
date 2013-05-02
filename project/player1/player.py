import common
import time
import random
import game_interface as game
import py.test

def get_move(view):
    if view.GetPlantInfo() == game.STATUS_NUTRITIOUS_PLANT:
        eat = True
    elif view.GetPlantInfo() == game.STATUS_UNKNOWN_PLANT:
        eat = random.choice([True,False])
    else:
        eat = False
    direction = random.choice([game.UP,game.DOWN,game.LEFT,game.RIGHT])
    time.sleep(0.1)
    return (direction, eat)
