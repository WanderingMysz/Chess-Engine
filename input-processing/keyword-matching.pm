package Keyword_Matching;

use warnings;
use strict;

use Exporter qw(import);
our @EXPORT_OK = qw(process_word);

use Text::Levenshtein qw(distance);

my @keywords = qw(ON FROM TO TAKES CAPTURES PROMOTES 
                  CHECK CHECKS CHECKMATE MATE);

sub process_word {
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
        return $keywords[$closest_idx] if $distances[closest_idx] <= 1;
        return undef;
    }
    return $keywords[$closest_idx] if $distances[$closest_idx] <= 2;
    return undef;
}

1;