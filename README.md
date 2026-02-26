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

# Step 3: Chess Engine - BonnieBot
The basic premise of the chess engine will be to recreate AlphaZero with 
insights from GPT training by having BonnieBot learn the rules of the game 
itself from Reinforcement Learning with AI Feedback. In essence, I will divide
model training into two stages: supervised and reinforcement. In the supervised
stage, BonnieBot will learn the rules of chess by being corrected by two
coaches: the chess library itself to ensure move validity and stockfish to
coach against outright blunders. Once BonnieBot has learned the rules of the
game, it can then progress to reinforcement learning by self play as with the
original AlphaZero model.

In the future, I could then have BonnieBot train future versions of itself, as
with GPT, to see if improvements can be made as a result.