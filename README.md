# Purpose
This is a personal project to better understand neural networks by constructing
my own chess engine from scratch. I am sure it will be a ton of work for
something that already exists, but I want to try my own ideas to gain a better
feeling for how things work.

One of these ideas is to explore how a lobalized structure influences
performance. That is, instead of having a singular neural network learn to play
chess, I will divide it into lobes that are each hyper-specialized (somehow)
which are then ensembled into a final engine.