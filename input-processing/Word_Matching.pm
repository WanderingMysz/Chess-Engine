package Word_Matching;

use warnings;
use strict;

use Exporter qw(import);
our @EXPORT_OK = qw(match_keyword match_piece 
                    @mating_terms @queenside_terms @kingside_terms @keywords 
                    %letter_codes %rev_letter_codes @piece_types);

use Text::Levenshtein qw(distance);

our @mating_terms       = qw(CHECK CHECKMATE MATE);
our @queenside_terms    = qw(QUEENSIDE LONG);
our @kingside_terms     = qw(KINGSIDE SHORT);
our @keywords           = (qw(ON FROM TO TAKES CAPTURES PROMOTES CASTLES),
                           @mating_terms, @queenside_terms, @kingside_terms);

our @piece_types = qw(KNIGHT BISHOP ROOK QUEEN KING PAWN);
our %letter_codes = (
    N => 'KNIGHT',
    B => 'BISHOP',
    R => 'ROOK',
    Q => 'QUEEN',
    K => 'KING',
    '' => 'PAWN',
);
our %rev_letter_codes = reverse %letter_codes;

sub match_keyword {
    my $word = uc(shift);

    # If the word is empty or a single char, return undefined
    return undef if length($word) <= 1;

    # Checks the Levenshtein distance of the word against each keyword.
    # If it is <=2, then it is considered a match.
    my @distances = distance($word, @keywords);
    my $closest_idx = 0;
    for my $i (1..$#distances) {
        $closest_idx = $i if $distances[$i] < $distances[$closest_idx];
    }

    # If the word is short, only match if it's a single typo.
    # If the word is not short, only match if it's 2 or fewer typos.
    # Otherwise it is undefined.
    if (length($word) <= 3) {
        return $keywords[$closest_idx] if $distances[$closest_idx] <= 1;
        return undef;
    }
    return $keywords[$closest_idx] if $distances[$closest_idx] <= 2;
    return undef;
}

sub match_piece {
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