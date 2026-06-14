import chess

def decode_action(action_index):
    """
    Translates a neural network output index (0-4095) back into a chess move.
    """
    from_square = action_index // 64
    to_square = action_index % 64
    
    move_string = chess.square_name(from_square) + chess.square_name(to_square)
    
    return move_string

def encode_move(move_string):
    """
    The reverse operation: translates a move (like 'e2e4') into an index (0-4095).
    v.v.v important for training the model
    """
    from_square = chess.parse_square(move_string[:2])
    to_square = chess.parse_square(move_string[2:4])
    
    action_index = (from_square * 64) + to_square
    return action_index


if __name__ == "__main__":
    test_move = "e2e4"
    idx = encode_move(test_move)
    decoded = decode_action(idx)
    print(f"Original: {test_move} -> Encoded Index: {idx} -> Decoded: {decoded}")