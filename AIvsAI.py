from env import env as game
import numpy as np
from VanilaMCTS import VanilaMCTS
import time

env = game.GameState()
state_size, win_mark = game.Return_BoardParams()
board_shape = [state_size, state_size]
game_board = np.zeros(board_shape, dtype=int)

game_end = False
whos_turn = {0: 'o', 1: 'x'}
mcts_players = {'o': VanilaMCTS(n_iterations=1500, depth=15, exploration_constant=100,
                                game_board=game_board, player='o'),
                'x': VanilaMCTS(n_iterations=1500, depth=15, exploration_constant=100,
                                game_board=game_board, player='x')}

current_player = 'o'

while not game_end:
    action_onehot = 0
    if current_player in mcts_players:
        mcts = mcts_players[current_player]
        best_action, best_q, depth = mcts.solve()
        action_onehot = np.zeros([state_size**2])
        action_onehot[best_action] = 1

    game_board, check_valid_position, win_index, turn = env.step(action_onehot)
    current_player = whos_turn[turn]

    if win_index != 0:
        game_board = np.zeros(board_shape, dtype=int)
        time.sleep(0.1)

        for player, mcts_player in mcts_players.items():
            mcts_player.tree = mcts_player._set_tictactoe(game_board, player)

    for player, mcts_player in mcts_players.items():
        mcts_player.tree[(0,)]['state'] = np.copy(game_board)

    action_idx = np.argmax(action_onehot)
    for player, mcts_player in mcts_players.items():
        mcts_player.tree[(0,)]['child'] = [
            a for a in mcts_player.tree[(0,)]['child'] if a != action_idx]
