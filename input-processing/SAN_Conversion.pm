package SAN_Conversion;

use warnings;
use strict;
use feature 'switch';
no warnings 'experimental::smartmatch';

use Exporter qw(import);
our @EXPORT_OK = qw(convert_to_SAN);

use Word_Matching qw(%rev_letter_codes);

sub convert_to_SAN {
    my ($move_record) = @_;
    my $ret_str = '';

    # Check for castling
    given ($move_record->{castle}) {
        when ("queenside")  { $ret_str = "O-O-O"; }
        when ("kingside")   { $ret_str = "O-O"; }
        default             { }
    }

    # Check for checks
    my $check       = '';
    if ($move_record->{check}) {
        $check .= '+' if ($move_record->{check} eq "check");
        $check .= '#' if ($move_record->{check} eq "checkmate");
    }

    # If castling occured, return with checking behavior
    if ($ret_str) { return $ret_str . $check; }

    # Otherwise, construct exhaustive SAN string
    my $piece       = $rev_letter_codes{$move_record->{type_from}};
    my $src         = $move_record->{from};
    my $dest        = $move_record->{to};
    my $capture     = ($move_record->{capture}) ? 'x' : '';

    my $promotion   = '';
    if ($move_record->{promotion}) {
        $promotion .= '=' . $rev_letter_codes{$move_record->{promotion}};
    }

    return join('', ($piece, $src, $capture, $dest, $promotion, $check));
}

1;
