import torch
import torch.nn as nn
import torch.nn.functional as F

import chess
import chess.engine

TOTAL_MOVES = 4096 # all possible piece movements
class BonnieBot(nn.Module):
    def __init__(self):
        super(BonnieBot, self).__init__()
        
        # Branch 1: Local Tactics 3x3
        self.local_tactics = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Branch 2: Mid-Range Tactics (Piece interactions) 5x5
        self.mid_tactics = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=1),
            nn.ReLU()
        )

        # Branch 3: Long-Range Dependencies (Bishop sniping, etc.) 8x8
        self.long_tactics = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=8),
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

def main():
    print("Queen's Gambit")

if __name__ == "__main__":
    main()