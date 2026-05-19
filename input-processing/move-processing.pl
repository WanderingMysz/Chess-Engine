#!/usr/bin/perl
use warnings;
use strict;

use FindBin qw($RealBin);
use lib $RealBin;

use Keyword_Matching ();
use Piece_Matching ();

my %move_record = map { uc($_) => 0 }
                    qw(typefrom typeto from to capture promotion check);

# Takes user move string, removes punctuation except for those necessary for
# SAN, then segments the move string into its constituent words
my $input = <STDIN>;
chomp($input);
$input =~ s/[^a-zA-Z0-9\s#+=]//g;

my @input = split(' ', $input);
my @input_uc = map { uc } @input;

my @keyword_indices = ();
my @piece_indices = ();

my $idx = 0;
for (@input_uc) {
    my $keyword = Keyword_Matching::process_word($_);
    my $piece = Piece_Matching::process_word($_);
    push @keyword_indices, $idx if defined $keyword;
    push @piece_indices, $idx if defined $piece;
    $idx++;
}

print("KEYWORDS FOUND\n");
for(@keyword_indices) {
    print("$input[$_] @ $_\n");
}
print("\nPIECES FOUND\n");
for(@piece_indices) {
    print("$input[$_] @ $_\n");
}