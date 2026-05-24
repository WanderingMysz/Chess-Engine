# Move Record Specs

## Format

| Datum             | Datatype      |
| ---               | ---           |
| Piece             | ChessPiece    |
| Source            | Coordinate    |
| Destination       | Coordinate    |
| Capture           | Flag          |
| Check             | Flag          |
| Checkmate         | Flag          |
| Castling          | Flag          |
| Piece-Secondary   | ChessPiece    |

## Rationale

Turn information, i.e. turn number and color, can be separately tracked in a
game record. This records all essential movement information, with `Piece`
included to allow for verification of `Source`->`Destination`, `Promotion`, and
`Castling` legality.

Castling could have been inferred using King movement since it is the only time
a king can move two squares. However, it is defined explicitly using a flag for
easier interpretability.

`Piece-Secondary` tracks promotions for pawns and the direction for castling.

Coordinates are used in place of Indices to promote human-readability of the
underlying XML.
