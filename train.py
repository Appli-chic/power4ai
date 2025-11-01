import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random
from dataclasses import dataclass
from game import ConnectFourGame
from model import ConnectFourNet
from utils import board_to_tensor, epsilon_greedy_action


GAMMA = 0.99
LEARNING_RATE = 0.0001
GRADIENT_CLIP_NORM = 1.0

INITIAL_EXPLORATION_RATE = 0.5
MIN_EXPLORATION_RATE = 0.1
EXPLORATION_DECAY = 0.995

SUMMARY_INTERVAL = 100
TARGET_NETWORK_UPDATE_INTERVAL = 100

REWARD_WIN = 1.0
REWARD_LOSS = -1.0
REWARD_DRAW = 0.0
REWARD_INTERMEDIATE = 0.0


@dataclass
class Transition:
    state: list
    action: int
    player: str
    next_state: list
    is_terminal: bool


def calculate_target_q_value(transition, winner, next_q_values, gamma):
    if transition.is_terminal:
        if winner == 'Draw':
            return REWARD_DRAW
        elif transition.player == winner:
            return REWARD_WIN
        else:
            return REWARD_LOSS
    else:
        max_next_q = next_q_values.max().item()
        return REWARD_INTERMEDIATE + gamma * max_next_q


def train_on_transition(model, target_model, optimizer, loss_fn, transition, winner, device, gamma):
    state_tensor = board_to_tensor(transition.state, transition.player, device)
    current_q_values = model(state_tensor)
    next_q_values = None

    if not transition.is_terminal:
        next_state_tensor = board_to_tensor(transition.next_state, transition.player, device)
        with torch.no_grad():
            next_q_values = target_model(next_state_tensor)

    target_q_value = calculate_target_q_value(transition, winner, next_q_values, gamma)

    target_q_values = current_q_values.clone().detach()
    target_q_values[transition.action] = target_q_value

    loss = loss_fn(current_q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
    optimizer.step()

    return loss.item()


def play_game(model, device, exploration_rate):
    game = ConnectFourGame()
    game.current_player = random.choice(['X', 'O'])
    transitions = []

    while not game.is_game_over:
        current_state = copy.deepcopy(game.board)
        current_player = game.current_player

        action = epsilon_greedy_action(model, current_state, current_player, exploration_rate, device)

        game.make_move(action)
        is_terminal = game.check_winner()

        transitions.append(
            Transition(
                state=current_state,
                action=action,
                player=current_player,
                next_state=copy.deepcopy(game.board),
                is_terminal=is_terminal
            )
        )

        if is_terminal:
            break

        game.switch_player()

    return transitions, game.winner


def print_game_result(winner):
    if winner == 'Draw':
        print(f"\nDRAW")
    else:
        print(f"\nPlayer {winner} WINS!")


def print_progress(game_num, wins_x, wins_o, draws, avg_loss, exploration_rate):
    total_games = game_num + 1
    win_rate = (wins_x + wins_o) / total_games * 100
    print(f"X wins: {wins_x} | O wins: {wins_o} | Draws: {draws} | Avg Loss: {avg_loss:.4f} | Win Rate: {win_rate:.1f}% | Exploration rate: {exploration_rate:.1f}%")


def train_self_play(num_games):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ConnectFourNet().to(device)
    target_model = ConnectFourNet().to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    exploration_rate = INITIAL_EXPLORATION_RATE
    wins_x = 0
    wins_o = 0
    draws = 0
    cumulative_loss = 0.0
    num_updates = 0

    for game_num in range(num_games):
        print(f"Game {game_num + 1}/{num_games}")

        transitions, winner = play_game(model, device, exploration_rate)

        if winner == 'Draw':
            draws += 1
        elif winner == 'X':
            wins_x += 1
        else:
            wins_o += 1

        print_game_result(winner)

        for transition in reversed(transitions):
            loss = train_on_transition(model, target_model, optimizer, loss_fn, transition, winner, device, GAMMA)
            cumulative_loss += loss
            num_updates += 1

        avg_loss = cumulative_loss / num_updates if num_updates > 0 else 0.0
        print_progress(game_num, wins_x, wins_o, draws, avg_loss, exploration_rate)

        if (game_num + 1) % SUMMARY_INTERVAL == 0:
            cumulative_loss = 0.0
            num_updates = 0

        if (game_num + 1) % TARGET_NETWORK_UPDATE_INTERVAL == 0:
            target_model.load_state_dict(model.state_dict())

        exploration_rate = max(MIN_EXPLORATION_RATE, exploration_rate * EXPLORATION_DECAY)

    torch.save(model.state_dict(), 'trained_model.pth')
    print("\nModel saved!")


if __name__ == '__main__':
    train_self_play(num_games=10_000)
