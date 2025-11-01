import torch
import os

from model import ConnectFourNet
from utils import model_select_action
from game import ConnectFourGame, get_player_move


def play_vs_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConnectFourNet().to(device)

    model_path = 'trained_model.pth'
    if not os.path.exists(model_path):
        print(f"Error: Trained model file '{model_path}' not found!")
        print("Please run train.py first to create a trained model.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded trained model from {model_path}")

    game = ConnectFourGame()

    print("\n=== CONNECT 4: HUMAN vs MODEL ===")
    print("You are X (goes first)")
    print("Model is O")
    print("Get 4 in a row to win!")

    game.print_board()

    while not game.is_game_over:
        if game.current_player == 'X':
            column = get_player_move(game.current_player)
            if not game.is_valid_move(column):
                print("Invalid move! Column is full or out of range. Try again.")
                continue
        else:
            print("Model is thinking...")
            column = model_select_action(model, game.board, game.current_player, device)
            if column is None:
                print("No valid moves available!")
                break
            print(f"Model chose column {column + 1}")

        if game.make_move(column):
            game.print_board()
            if not game.check_winner():
                game.switch_player()

    if game.winner == 'X':
        print("You win! Congratulations!")
    elif game.winner == 'O':
        print("Model wins!")
    elif game.winner == 'Draw':
        print("It's a draw! The board is full.")


if __name__ == "__main__":
    print("\n=== Connect Four AI ===")
    play_vs_model()
