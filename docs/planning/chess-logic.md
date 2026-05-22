# What must be done

## Move Information

| category | type |
| --- | --- |
| turn # | numeric |
| color | flag |
| piece type | categorical |
| source | numeric |
| destination | numeric |
| capture | flag |
| promotion | categorical |
| check(mate) | categorical |

Castling can be implicit in a king's two square movement or in using the
promotion type for king moves to indicate direction (queen or king).

User input is processed as Standard Algebraic Notation (SAN). SAN explicitly
contains all the information in the table except for source indices if there is
no ambiguity.

However, a good chess program must validate what is provided: promotion, en
passant, check, etc.

## Logical Flow

1. User input read in.
2. SAN is interpreted, source index calculated based on present board state.
3. If all seems good, the current board state is saved and the move is made.
4. A check detector is run. If the current player's king is in check, then the
    move is rolled back. Otherwise, it is allowed.
5. Check-checkmate detection is then run on the opponent king. This is then
    reported to the other player and the game ended if checkmate.
