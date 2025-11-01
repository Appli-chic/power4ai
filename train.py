import torch
import torch.nn as nn
import torch.optim as optim
import copy
from dataclasses import dataclass
from game import ConnectFourGame
from model import ConnectFourNet
from utils import board_to_tensor, model_select_action


@dataclass
class Move:
    board: list
    action: int
    player: str


def train_self_play(num_games):
    model = ConnectFourNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    wins_x = 0
    wins_o = 0
    draws = 0

    for game_num in range(num_games):
        print(f"Game {game_num + 1}/{num_games}")

        game = ConnectFourGame()
        moves = []

        while not game.is_game_over:
            board_copy = copy.deepcopy(game.board)
            player = game.current_player

            # To improve learning, add randomness (epsilon-greedy), epsilon_greedy_action in utils.py:
            # - Sometimes pick random moves to explore new strategies
            # - Gradually reduce randomness as the model improve
            action = model_select_action(model, board_copy, player)

            moves.append(
                Move(
                    board=board_copy,
                    action=action,
                    player=player
                )
            )

            game.make_move(action)
            game.print_board()
            print(f"Player {player} chose column {action + 1}")

            if game.check_winner():
                break

            game.switch_player()

        winner = game.winner

        if winner == 'Draw':
            draws += 1
            print(f"\nDRAW")
        else:
            if winner == 'X':
                wins_x += 1
            else:
                wins_o += 1
            print(f"\nPlayer {winner} WINS!")

        for move in moves:
            board_tensor = board_to_tensor(move.board, move.player)

            model_prediction = model(board_tensor)
            corrected_model_prediction = model_prediction.clone().detach()

            if winner == 'Draw':
                corrected_model_prediction[move.action] = 0
            elif move.player == winner:
                corrected_model_prediction[move.action] = 1
            else:
                corrected_model_prediction[move.action] = -1

            loss = loss_fn(model_prediction, corrected_model_prediction)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"X wins: {wins_x} | O wins: {wins_o} | Draws: {draws}")

    torch.save(model.state_dict(), 'trained_model.pth')
    print("\nModel saved!")


if __name__ == '__main__':
    train_self_play(num_games=10_000)
