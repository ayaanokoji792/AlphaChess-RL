import chess
from encoder import board_to_tensor
from model import ChessNet

def main():
    board = chess.Board()
    model = ChessNet()
    
    print("Initial Board State:")
    print(board)
    print("\n-------------------\n")
    
    state_tensor = board_to_tensor(board)
    policy_logits, value = model(state_tensor)
    
    print(f"Network's Evaluation (Value Head): {value.item():.3f}")
    print("(-1 = Black leading, 0 = Draw/Even, 1 = White leading)")

if __name__ == "__main__":
    main()