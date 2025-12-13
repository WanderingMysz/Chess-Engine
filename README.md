# Purpose
This is a personal project to better understand neural networks by constructing
my own chess engine from scratch. I am sure it will be a ton of work for
something that already exists, but I want to try my own ideas to gain a better
feeling for how things work.

One of these ideas is to explore how a lobalized structure influences
performance. That is, instead of having a singular neural network learn to play
chess, I will divide it into lobes that are each hyper-specialized (somehow)
which are then ensembled into a final engine.

UPDATE: I am expanding this to help refine other development skills. These are
defined in the Design Plan below.

# Design Plan
1. Create a basic text-based chess game in C. It will accept CLI inputs denoting
    player moves, correcting the player when an illegal move is made.
    - Basic ASCII rendering
    - Informs player when they have made an illegal move
    - Automatically determines check / checkmate
    - Terminal clears between moves for seamless look

    EXTENSION: Improve the board from basic ASCII to stylized ASCII art

2. Design a TUI using Rust or Go.

3. Develop the chess engine in PyTorch.

4. Design a GUI for playing with the chess engine. Language TBD.