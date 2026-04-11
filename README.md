# Chess from Scratch

## Overview

### Purpose

This a personal project for practice in working across an entire tech stack. It
uses a separate language for each core element purposely to force me to work
outside my comfort zone and take advantage of the inherent strengths of each
language.

### Why Chess?

Aside from personal interest in the game, Chess has straightforward rules and
is easily interpretable. Representationally, it is simple. All it needs is an
8x8 checkerboard and the pieces which can be done in a CLI. Yet, it also it a
benchmark for computer ability going back decades, from early attempts at chess
engines to Deep Blue to AlphaZero.

Plus, at the end of the day, it'll be something I can actually play.

### Scope

Currently, the project is focused on developing just the CLI application in `C`
and an engine, BonnieBot, in `PyTorch`. I don't want to spread myself too thin
jumping from element to element and language to language. The focus is on
practical depth. However, I am designing with extendability in mind. This is a
sandbox which I can constantly add to to practice. Those ideas are described in
the tech stack below.

## Proposed Tech Stack

### Chess Engine

| Component | Language / Framework | Purpose |
| --- | --- | --- |
| BonnieBot | Python, PyTorch | CNN-based chess engine loosely modeled on AlphaZero |
| FreddyBot | Python, JAX | Vision Transformer-based engine |

### Chess Logic & CLI

| Component | Language | Purpose |
| --- | --- | --- |
| Move validation & board representation | C | Core chess logic library |
| CLI application | C | CLI chess client using Unicode representations |

### Data Pipeline

| Component | Language | Purpose |
| --- | --- | --- |
| PGN parser | AWK | Extracts and cleans move data from Lichess PGN files |
| Dataset validation | COBOL | Validates move files before they are used for engine training |
| Statistics | COBOL | Reports on batch game statistics, such as average game length, etc. |
| Training data encoder | Python | Encodes move records into tensors for model training |

### Interfaces

| Component | Language / Framework | Purpose |
| --- | --- | --- |
| TUI application | Go | Terminal interface |
| Web interface | Next.js | Browser-based interface |

### Infrastructure

| Component | Tool | Purpose |
| --- | --- | --- |
| Containerization | Docker, Docker Compose | Containerize the various components described above |
| Game history | PostgreSQL | Database of completed games and engine reviews |

## Current Progress

### CLI

A basic Unicode chess application for the terminal. It allows two users to play
against each other by inputting moves in Standard Algebraic Notation (SAN). It
is also self-contained with all the necessary logic baked in.

Written in `C`, it will be extended to Python for use in engine training as a 
`ctype`. Unit testing performed with `Unity`.

#### Basic Functionality

- [x] Basic ASCII rendering of board
- [x] Accepts SAN-formatted user input
- [ ] Validates moves, reprompting user if move invalid
- [ ] Automatically checks for check / checkmate
- [ ] Terminal clears between inputs, creating seamless look while maintaining
        history

#### Extended Functionality

- [ ] Exports / stores game data
- [ ] Accepts word-formatted user input, e.g. "Knight takes f4"

### The Engines

Loosely based on AlphaZero, BonnieBot uses a `PyTorch` CNN to interpret
gamestate information passed as planes with improvement through self play.
However, it also learns the rules of the game through observation of existing
games (culled from Lichess), rather than be explicitly taught.

FreddyBot is intended to be similar in design but using `JAX`
Vision-Transformers. This is for more exposure to ViTs and to compare
performance between the two architectures on this task.

#### BonnieBot

- [x] CNN architecture designed and implemented
- [ ] Learns rules through observation
- [ ] Improves through self-play

#### FreddyBot

- [ ] ViT architecture designed and implemented
- [ ] Learns rules through observation
- [ ] Improves through self-play

### Data Pipeline

How the raw game data is handled. This includes PGN files from external sources
as well as local game representations.

#### Basic Functionality

- [x] Processes external PGN files into more structured format

#### Extended Functionality

- [ ] Provides batch game statistics
