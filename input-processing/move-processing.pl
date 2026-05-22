#!/usr/bin/perl
use warnings;
use strict;

use FindBin qw($RealBin);
use lib $RealBin;

use Word_Matching qw(match_keyword match_piece);
use Keyword_Processing qw(process_keyword);
use SAN_Conversion qw(convert_to_SAN);

my %move_record = map { lc($_) => "" }
    qw(type_from type_to from to capture castle promotion check);

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

# Useful record for processing, collates information on specific pieces
my @piece_info = ();
while (my ($piece_idx, $piece_type) = each %found_pieces) {
    push @piece_info, { idx => $piece_idx, 
                        type => $piece_type, 
                        role => "", 
                        coord => "" };
}

# Process all keywords, extracting move and piece information
while (my ($keyword_idx, $keyword) = each %found_keywords) {
    process_keyword($keyword_idx, \%found_keywords, \%found_coords, 
                    \%found_pieces, \@piece_info, \%move_record);
}

# Process piece information to fill holes in the move record
foreach my $piece (@piece_info) {
    my $coord   = $piece->{coord};
    my $role    = $piece->{role};

    # If the role is explicitly known, update accordingly
    # Otherwise, try to match using coordinates
    if ($role eq "attack") {
        $move_record{type_from} = $piece->{type};
        $move_record{from} = $piece->{coord};
    } elsif ($role eq "defend") {
        $move_record{type_to} = $piece->{type};
        $move_record{to} = $piece->{coord};
    } elsif ($coord) {
        if ($coord eq $move_record{from}) {
            $move_record{type_from} = $piece->{type};
        } elsif ($coord eq $move_record{to}) {
            $move_record{type_to} = $piece->{type};
        }
    }
}

my $SAN_string = convert_to_SAN(\%move_record);
print ("$SAN_string\n");
