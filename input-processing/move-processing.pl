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

print("KEYWORDS FOUND\n");
for(keys %found_keywords) {
    print("$input[$_] @ $_\n");
}
print("\nPIECES FOUND\n");
for(keys %found_pieces) {
    print("$input[$_] @ $_\n");
}