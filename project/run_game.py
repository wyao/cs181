import common
import player1.player
import player2.player
import game_interface
import random
import signal
import sys
import time
import traceback
import cPickle
from optparse import OptionParser

class TimeoutException(Exception):
  def __init__(self):
    pass

def get_move(view, cmd, options, player_id):
  def timeout_handler(signum, frame):
    raise TimeoutException()
  signal.signal(signal.SIGALRM, timeout_handler)
  # signal.alarm(1)
  try: 
    (mv, eat) = cmd(view,options)
    # Clear the alarm.
    signal.alarm(0)
  except TimeoutException:
    # Return a random value
    # Should probably log this to the interface
    (mv, eat) = (random.randint(0, 4), False)
    error_str = 'Error in move selection (%d).' % view.GetRound()
    if options.display:
      game_interface.curses_debug(player_id, error_str)
    else:
      print error_str
  return (mv, eat)

def run(options):
  game = game_interface.GameInterface(options.plant_bonus,
                                      options.plant_penalty,
                                      options.observation_cost,
                                      options.starting_life,
                                      options.life_per_turn)
  player1_view = game.GetPlayer1View()
  player2_view = game.GetPlayer2View()

  if options.display:
    if game_interface.curses_init() < 0:
      return
    game_interface.curses_draw_board(game)
  
  # Keep running until one player runs out of life.
  while True:
    (mv1, eat1) = get_move(player1_view, player1.player.get_move, options, 1)
    if options.train != 0:
      (mv2, eat2) = get_move(player2_view, player2.player.get_move, options, 2)
    else:
      (mv2, eat2) = get_move(player2_view, common.get_move, options, 2)
    game.ExecuteMoves(mv1, eat1, mv2, eat2)
    if options.display == 1:
      game_interface.curses_draw_board(game)
      game_interface.curses_init_round(game)
    elif options.display == 2:
      print mv1, eat1, mv2, eat2
      print player1_view.GetLife(), player2_view.GetLife()
    # Check whether someone's life is negative.
    l1 = player1_view.GetLife()
    l2 = player2_view.GetLife()
  
    if l1 <= 0 or l2 <= 0:
      # Export Q
      if options.q_out != None:
        # print "Exporting Q to", options.q_out
        f = open(options.q_out, "w")
        cPickle.dump(player1.player.Q, f)
        f.close()
      # Export neural network weights
      if options.train == 1:
        # print "Exporting weights to", options.out_file
        player2.player.network.ExportWeights(options.out_file)
      elif options.train == 2:
        print player2.player.correct / player2.player.instances
      elif options.train == 3:
        # print "Exporting plants to", options.out_file
        player2.player.ExportPlants(options.out_file)
      if options.display:
        winner = 0
        if l1 < l2:
          winner = 2
        else:
          winner = 1
        game_interface.curses_declare_winner(winner)
      else:
        if l1 == l2:
          print 'Tie, remaining life: %d v. %d' % (l1, l2)
        elif l1 < l2:
          print 'Player 2 wins: %d v. %d' % (l1, l2)
        else:
          print 'Player 1 wins: %d v. %d' % (l1, l2)
      # Wait for input
      sys.stdin.read(1)
      if options.display:
        game_interface.curses_close()
      break

def main(argv):
  parser = OptionParser()
  parser.add_option("-d", action="store", dest="display", default=1, type=int,
                    help="whether to display the GUI board")
  parser.add_option("--plant_bonus", dest="plant_bonus", default=20,
                    help="bonus for eating a nutritious plant",type=int)
  parser.add_option("--plant_penalty", dest="plant_penalty", default=10,
                    help="penalty for eating a poisonous plant",type=int)
  parser.add_option("--observation_cost", dest="observation_cost", default=1,
                    help="cost for getting an image for a plant",type=int)
  parser.add_option("-s", "--starting_life", dest="starting_life", default=100,
                    help="starting life",type=int)
  parser.add_option("--life_per_turn", dest="life_per_turn", default=1,
                    help="life spent per turn",type=int)
  parser.add_option("--hidden", type=int, default=15)
  parser.add_option("--train", type=int, default=0)
  parser.add_option("--in_file", type="string", default="weight.txt")
  parser.add_option("--out_file", type="string", default="weight.txt")
  parser.add_option("--q_in", type="string", default=None)
  parser.add_option("--q_out", type="string", default=None)
  (options, args) = parser.parse_args()

  try:
    run(options)
  except KeyboardInterrupt:
    if options.display:
      game_interface.curses_close()
  except:
    game_interface.curses_close()
    traceback.print_exc(file=sys.stdout)

if __name__ == '__main__':
  main(sys.argv)
