class ConnectFourGame:
    def __init__(self):
        self.board = [[' ' for _ in range(7)] for _ in range(6)]
        self.current_player = 'X'
        self.is_game_over = False
        self.winner = None
        self.move_history = []

    def reset(self):
        self.board = [[' ' for _ in range(7)] for _ in range(6)]
        self.current_player = 'X'
        self.is_game_over = False
        self.winner = None
        self.move_history = []

    def make_move(self, column):
        if not self.is_valid_move(column):
            return False

        for row in range(5, -1, -1):
            if self.board[row][column] == ' ':
                self.board[row][column] = self.current_player
                self.move_history.append(column)
                return True
        return False

    def is_valid_move(self, column):
        return 0 <= column < 7 and self.board[0][column] == ' '

    def check_winner(self):
        piece = self.current_player

        for row in range(6):
            for col in range(4):
                if all(self.board[row][col+i] == piece for i in range(4)):
                    self.is_game_over = True
                    self.winner = piece
                    return True

        for row in range(3):
            for col in range(7):
                if all(self.board[row+i][col] == piece for i in range(4)):
                    self.is_game_over = True
                    self.winner = piece
                    return True

        for row in range(3):
            for col in range(4):
                if all(self.board[row+i][col+i] == piece for i in range(4)):
                    self.is_game_over = True
                    self.winner = piece
                    return True

        for row in range(3):
            for col in range(3, 7):
                if all(self.board[row+i][col-i] == piece for i in range(4)):
                    self.is_game_over = True
                    self.winner = piece
                    return True

        if self.is_full():
            self.is_game_over = True
            self.winner = 'Draw'
            return True

        return False

    def is_full(self):
        return all(self.board[0][col] != ' ' for col in range(7))

    def switch_player(self):
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def print_board(self):
        print("\n")
        print("  1   2   3   4   5   6   7")

        for i, row in enumerate(self.board):
            print("│", end="")
            for cell in row:
                print(f" {cell} ", end="│")
            print()
            if i < len(self.board) - 1:
                print("├───┼───┼───┼───┼───┼───┼───┤")
            else:
                print("└───┴───┴───┴───┴───┴───┴───┘")
        print()

    def get_board(self):
        return self.board

    def get_status(self):
        return self.is_game_over, self.winner


def get_player_move(player):
    while True:
        try:
            column = int(input(f"Player {player}, choose column (1-7): ")) - 1
            return column
        except ValueError:
            print("Invalid input! Please enter a number between 1 and 7.")


def is_valid_move(board, column):
    return 0 <= column < 7 and board[0][column] == ' '
