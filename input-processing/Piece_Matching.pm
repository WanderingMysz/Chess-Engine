package Piece_Matching;

use warnings;
use strict;

use Exporter qw(import);
our @EXPORT_OK = qw(process_word);

use Text::Levenshtein qw(distance);

my %letter_codes = (
    N => 'KNIGHT',
    B => 'BISHOP',
    R => 'ROOK',
    Q => 'QUEEN',
    K => 'KING',
    P => 'PAWN',
);

my @piece_types = qw(KNIGHT BISHOP ROOK QUEEN KING PAWN);

sub process_word {
    my $word = uc(shift);

    # If word is empty, then it refers to a pawn by convention.
    # If it is just a single letter, match per the letter codes.
    return 'PAWN' if length($word) == 0;
    return $letter_codes{$word} if length($word) == 1;

    # TODO: What about when it matches two? Like Kiok : King | Rook

    # Checks the Levenshtein distance of the word against each piece type.
    # If it is <=2, then it is considered a match.
    my @distances = distance($word, @piece_types);
    my $closest_idx = 0;
    for my $i (1..$#distances) {
        $closest_idx = $i if $distances[$i] < $distances[$closest_idx];
    }
    return $piece_types[$closest_idx] if $distances[$closest_idx] <= 2;
    return undef;
}

1;