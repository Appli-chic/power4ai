import torch
from game import is_valid_move


def board_to_tensor(board, player_piece):
    opponent_piece = 'O' if player_piece == 'X' else 'X'
    tensor_board = []

    for row in board:
        for cell in row:
            if cell == player_piece:
                tensor_board.append(1.0)
            elif cell == opponent_piece:
                tensor_board.append(-1.0)
            else:
                tensor_board.append(0.0)

    return torch.FloatTensor(tensor_board)


def get_valid_moves(board):
    return [col for col in range(7) if is_valid_move(board, col)]


def model_select_action(model, board, player_piece):
    valid_moves = get_valid_moves(board)
    if not valid_moves:
        return None

    tensor_board = board_to_tensor(board, player_piece)

    with torch.no_grad():
        q_values = model(tensor_board).numpy()

    for col in range(7):
        if col not in valid_moves:
            q_values[col] = -float('inf')

    return int(q_values.argmax())
