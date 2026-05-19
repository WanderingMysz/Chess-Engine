#!/usr/bin/perl
use warnings;
use strict;

use FindBin qw($RealBin);
use lib $RealBin;

use Word_Matching qw(match_keyword match_piece);

my %move_record = map { uc($_) => 0 }
                    qw(typefrom typeto from to capture promotion check);

# Takes user move string, removes punctuation except for those necessary for
# SAN, then segments the move string into its constituent words
chomp(my $input = <STDIN>);
$input =~ s/[^a-zA-Z0-9\s#+=]//g;

my @input = split(' ', $input);
my @input_uc = map { uc } @input;

# Finds keywords, pieces, and coordinates, storing their standardized versions
my (%found_keywords, %found_pieces, %found_coords);

my $idx = 0;
foreach (@input_uc) {
    my $keyword = match_keyword($_);
    my $piece = match_piece($_);

    $found_keywords{$idx} = $keyword if defined $keyword;
    $found_pieces{$idx} = $piece if defined $piece;
    $found_coords{$idx} = lc($_) if $_ =~ /^[A-Z][0-9]$/;
    $idx++;
}

# There should be at most four pieces in a sensible string:
# Attacker, Defender, Promotion, Checks (King)
die "Malformed input string. More than 4 pieces present.\n" 
    if scalar %found_pieces > 4;

# If words are assigned to multiple types, malformed input
my $duplicate_idx = 0;
sub compare_hashes {
    my $hash1_ptr = shift;
    my $hash2_ptr = shift;

    foreach my $hash1_idx (keys %$hash1_ptr) {
        foreach my $hash2_idx (keys %$hash2_ptr) {
            $duplicate_idx++ if $hash1_idx == $hash2_idx;
        }
    }
}
compare_hashes(\%found_keywords, \%found_pieces);
compare_hashes(\%found_keywords, \%found_coords);
compare_hashes(\%found_coords, \%found_pieces);
if ($duplicate_idx) {
    die "Ambiguity exists in one or more words. Please input move again.\n";
}