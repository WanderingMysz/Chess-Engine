package san_conversion;

use warnings;
use strict;
use feature 'switch';
no warnings 'experimental::smartmatch';

use Exporter qw(import);
our @EXPORT_OK = qw(convert_to_SAN);

use word_matching qw(%rev_letter_codes);

sub convert_to_SAN {
    my ($move_record) = @_;
    my $ret_str = '';

    # Check for castling
    if ($move_record->{castling_flag}) {
        given ($move_record->{opt_info}) {
            when ("QUEEN")  { $ret_str = "O-O-O"; }
            when ("KING")   { $ret_str = "O-O"; }
            default         { }
        }
    }

    # Check for checks
    my $check       = '';
    if ($move_record->{check_flag}) {
        $check .= ($move_record->{checkmate_flag}) ? '#' : '+';
    }

    # If castling occured, return with checking behavior
    if ($ret_str) { return $ret_str . $check; }

    # Otherwise, construct exhaustive SAN string
    my $piece       = $rev_letter_codes{$move_record->{piece_attack} // ''} 
                        // '';
    my $src         = $move_record->{src_square} // '';
    my $dest        = $move_record->{dest_square} // '';
    my $capture     = ($move_record->{capture_flag}) ? 'x' : '';

    my $promotion   = '';
    if ($move_record->{promotion_flag}) {
        $promotion .= '=' . $rev_letter_codes{$move_record->{opt_info}};
    }

    return join('', ($piece, $src, $capture, $dest, $promotion, $check));
}

1;
