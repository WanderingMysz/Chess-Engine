# Move Record Specs

## Format

| Datum             | Datatype      |
| ---               | ---           |
| Piece_Attack      | ChessPiece    |
| Piece_Defend      | ChessPiece    |
| Opt_Info          | ChessPiece    |
| Src_Square        | Coordinate    |
| Dest_Square       | Coordinate    |
| Capture_Flag      | Flag          |
| Check_Flag        | Flag          |
| Checkmate_Flag    | Flag          |
| Castling_Flag     | Flag          |
| Promotion_Flag    | Flag          |

## Rationale

Turn information, i.e. turn number and color, can be separately tracked in a
game record. This records all essential movement information, with
`Piece_Attack` included to allow for verification of `Src_Square`->
`Dest_Square` and `Promotion` / `Castling` legality. `Piece_Defend` is required
for proper SAN notation in Perl, etc. Everything is encoded in string format,
meanwhile to ensure interoperability between modules (mostly because of the
presence of COBOL).

Castling could have been inferred using King movement since it is the only time
a king can move two squares. However, it is defined explicitly using a flag for
easier interpretability.

`Opt_Info` tracks promotions for pawns and the direction for castling.
`Coordinate` is used in place of Indices to promote human-readability of the
underlying XML.
