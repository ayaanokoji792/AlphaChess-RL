import chess
import numpy as np
import torch

def board_to_tensor(board):
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    piece_map = board.piece_map()
    
    for square, piece in piece_map.items():
        row = chess.square_rank(square)
        col = chess.square_file(square)
        
        piece_type = piece.piece_type - 1 
        piece_color = 0 if piece.color == chess.WHITE else 6
        channel = piece_type + piece_color
        
        tensor[channel][row][col] = 1.0
        
    return torch.tensor(tensor).unsqueeze(0)