from model import ConnectFourNet
from utils import model_select_action
from game import ConnectFourGame, get_player_move


def play_vs_model():
    model = ConnectFourNet()
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
            column = model_select_action(model, game.board, game.current_player)
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


def train_self_play():
    model = ConnectFourNet()
    game = ConnectFourGame()

    print("\n=== SELF-PLAY TRAINING ===")
    print("Model plays as both X and O")
    print("Watching AI play against itself...\n")

    game.print_board()

    while not game.is_game_over:
        print(f"{game.current_player} is thinking...")
        column = model_select_action(model, game.board, game.current_player)

        if column is None:
            print("No valid moves available!")
            break

        print(f"{game.current_player} chose column {column + 1}")

        if game.make_move(column):
            game.print_board()
            if not game.check_winner():
                game.switch_player()

    print("\n=== GAME OVER ===")
    if game.winner == 'X':
        print("X wins!")
    elif game.winner == 'O':
        print("O wins!")
    elif game.winner == 'Draw':
        print("Draw! The board is full.")


if __name__ == "__main__":
    print("\n=== Connect Four AI ===")
    print("1. Play against the model")
    print("2. Watch AI self-play training")

    choice = input("\nChoose option (1 or 2): ").strip()

    if choice == "1":
        play_vs_model()
    elif choice == "2":
        train_self_play()
    else:
        print("Invalid choice!")
