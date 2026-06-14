import torch
import torch.optim as optim
import chess
import random
from encoder import board_to_tensor
from model import ChessNet
from decoder import encode_move
from train import train_step

def run_mini_training_session():
    print("Initializing AlphaChess Training Engine...")
    model = ChessNet()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    board = chess.Board()
    
    states_collected = []
    target_policies_collected = []
    
    print("\n--- Phase 1: Simulating Self-Play Game ---")
    move_count = 0
    # Play 10 moves or until game ends
    while not board.is_game_over() and move_count < 10:
        state_tensor = board_to_tensor(board)
        states_collected.append(state_tensor.squeeze(0)) # Remove batch dim for stacking later
        

        legal_moves = list(board.legal_moves)
        chosen_move = random.choice(legal_moves)
        
        #creating a mock MCTS policy target for the chosen move
        target_policy = torch.zeros(4096)
        chosen_idx = encode_move(chosen_move.uci())
        target_policy[chosen_idx] = 1.0
        target_policies_collected.append(target_policy)
        

        board.push(chosen_move)
        move_count += 1
        print(f"Move {move_count}: {chosen_move.uci()}")

    print(f"\nSelf-play session finished. Generated {len(states_collected)} board states.")

    print("\n--- Phase 2: Running Optimization Step ---")

    state_batch = torch.stack(states_collected)
    target_policies = torch.stack(target_policies_collected)
    
    
    #assuming a mock game outcome of white winning for simplicity
    target_values = torch.ones(len(states_collected))

    #BACKPROPAGATIONNNN
    total_loss, v_loss, p_loss = train_step(model, optimizer, state_batch, target_policies, target_values)
    
    print("\n==========================================")
    print("TRAINING STATUS: SUCCESS")
    print(f"Total Loss: {total_loss:.4f}")
    print(f" -> Value Head Loss: {v_loss:.4f}")
    print(f" -> Policy Head Loss: {p_loss:.4f}")
    print("==========================================")
    print("Model parameters updated successfully.")

if __name__ == "__main__":
    run_mini_training_session()