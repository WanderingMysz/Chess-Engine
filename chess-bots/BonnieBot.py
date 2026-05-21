import torch
import torch.nn as nn
import torch.nn.functional as F

import chess
import chess.engine

import numpy as np

TOTAL_MOVES = 4096 # all possible piece movements
PIECE_TYPES = 12 # 12 in standard chess, leaves flexibility for fairy chess
class BonnieBot(nn.Module):
    def __init__(self):
        super(BonnieBot, self).__init__()
        
        # Branch 1: Local Tactics 3x3
        self.local_tactics = nn.Sequential(
            nn.Conv2d(PIECE_TYPES, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Branch 2: Mid-Range Tactics (Piece interactions) 5x5
        self.mid_tactics = nn.Sequential(
            nn.Conv2d(PIECE_TYPES, 64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=1),
            nn.ReLU()
        )

        # Branch 3: Long-Range Dependencies (Bishop sniping, etc.) 8x8
        self.long_tactics = nn.Sequential(
            nn.Conv2d(PIECE_TYPES, 64, kernel_size=8),
            nn.ReLU()
        )

        # local + mid + long
        flattened_size = (64 * 8 * 8) + (64 * 6 * 6) + (64 * 1 * 1)

        self.fc1 = nn.Linear(flattened_size, 1024)
        self.decision = nn.Linear(1024, TOTAL_MOVES)

    def forward(self, x):
        out_local = self.local_tactics(x)
        out_mid = self.mid_tactics(x)
        out_long = self.long_tactics(x)

        out_local = out_local.view(out_local.size(0), -1)
        out_mid = out_mid.view(out_mid.size(0), -1)
        out_long = out_long.view(out_long.size(0), -1)

        combined = torch.cat((out_local, out_mid, out_long), dim=1)

        x = F.relu(self.fc1(combined))
        decision = self.decision(x)

        return decision
    

def extract_current_state(board):
    """
    Takes a board state and returns it as a twelve-plane tensor: one plane
    per piece-type per color

    Args: 
        board: current board state
    
    Returns:
        multiplanar tensor, one plane per piece-type per color, 8x8
    """
    # 12 planes, 8x8 board
    tensor = np.zeros((PIECE_TYPES,8,8), dtype=np.int8)

    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
              chess.ROOK, chess.QUEEN, chess.KING]
    piece_to_type = {type: i for i, type in enumerate(piece_types)}

    if (len(piece_types) != PIECE_TYPES // 2):
        raise Exception("The number of piece types enumerated does not equal" +
                        " what is defined globally.")

    for square in chess.SQUARES:
        piece = board.piece_at(square)

        if piece: # ignore empty squares
            row, col = chess.square_rank(square), chess.square_file(square)

            plane = piece_to_type[piece.piece_type]
            plane = plane if piece.color == chess.WHITE else plane + PIECE_TYPES

            tensor[plane][row][col] = 1.0

    return torch.from_numpy(tensor)
    

def main():
    print("Queen's Gambit")

if __name__ == "__main__":
    main()