for i in `seq 1 5000`
do
  echo -ne '\n' | python run_game.py -d0 --in_file player1/e5_l03.weights --q_out player1/1.q --q_in player1/1.q
done
