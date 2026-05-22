package Keyword_Processing;

use warnings;
use strict;
use feature 'switch';
no warnings 'experimental::smartmatch';

use Exporter qw(import);
our @EXPORT_OK = qw(process_keyword);

use Word_Matching qw(@queenside_terms @kingside_terms);

our ($idx, $keywords, $coords, $pieces, $piece_info, $move_record);
my @TO_EXCEPTIONS = qw(PROMOTES CASTLES);

# Since @piece_info is small (<=4), simply iterate over whole array to update
sub _update_piece_info {
    my ($piece_idx, $datatype, $datum) = @_;

    if (defined $datum) {
        foreach (@$piece_info) {
            if ($_->{"idx"} == $piece_idx) {
                $_->{$datatype} = $datum;
                return 0;
            }
        }
    }
    return 1;
}

sub _update_move_record {
    my ($datatype, $datum) = @_;

    if (defined $datum) {
        $move_record->{$datatype} = $datum;
        return 0;
    }
    return 1;
}

# ------
# Update information based on specific keywords present
# ------

# piece ON coord
sub _on {
    my $coord = $coords->{$idx+1};
    _update_piece_info($idx-1, "coord", $coord);
}
# piece / coord TO coord
sub _to {
    # Ignore if previous word is "promotes"
    if (grep {$_ eq $keywords->{$idx-1} } @TO_EXCEPTIONS) { return 0; }

    my $piece       = $pieces->{$idx-1};
    my $coord_from  = $coords->{$idx-1};
    my $coord_to    = $coords->{$idx+1};

    my $ret = 0;
    if (defined $piece) {
        $ret |= _update_piece_info($idx-1, "role", "attack");
        $ret |= _update_move_record("type_from", $piece);
    } elsif (defined $coord_from) {
        $ret |= _update_move_record("from", $coord_from);
    } else { return 1; }

    $ret |= _update_move_record("to", $coord_to);
    return $ret;
}
# piece FROM coord
sub _from {
    my $piece       = $pieces->{$idx-1};
    my $coord_from  = $coords->{$idx+1};

    my $ret = 0;
    $ret |= _update_piece_info($idx-1, "role", "attack");
    $ret |= _update_move_record("type_from", $piece);
    $ret |= _update_move_record("from", $coord_from);
    return $ret;
}
# piece / coord CAPTURE piece / coord
sub _capture {
    my $piece_from  = $pieces->{$idx-1};
    my $piece_to    = $pieces->{$idx+1};
    my $coord_from  = $coords->{$idx-1};
    my $coord_to    = $coords->{$idx+1};

    my $ret = 0;
    if (defined $piece_from) {
        $ret |= _update_piece_info($idx-1, "role", "attack");
        $ret |= _update_move_record("type_from", $piece_from);
    } elsif (defined $coord_from) {
        $ret |= _update_move_record("from", $coord_from);
    } else { return 1; }

    if (defined $piece_to) {
        $ret |= _update_piece_info($idx+1, "role", "defend");
        $ret |= _update_move_record("type_to", $piece_to);
    } elsif (defined $coord_to) {
        $ret |= _update_move_record("to", $coord_to);
    } else { return 1; }

    $ret |= _update_move_record("capture", "y");

    return $ret;
}
# (coord) PROMOTE (TO) piece
sub _promote {
    my $piece = ($keywords->{$idx+1} eq "TO") ? $pieces->{$idx+2} 
                                              : $pieces->{$idx+1};
    my $coord = $coords->{$idx-1};

    my $ret = 0;
    # $ret |= _update_move_record("from", $coord) if (defined $coord);
    $ret |= _update_move_record("promotion", $piece);
    return $ret;
}
# CHECK
sub _check {
    _update_move_record("check", "check");
}
# MATE
sub _mate {
    _update_move_record("check", "checkmate");
}
# CASTLE {KINGSIDE / QUEENSIDE}
sub _castle {
    my $direction = ($keywords->{$idx+1} eq "TO") ? $keywords->{$idx+2} 
                                                  : $keywords->{$idx+1};
    $direction = uc ($direction);

    if (!defined $direction) { return 1 };
    if (grep {$direction eq $_} @queenside_terms) {
        _update_move_record("castle", "queenside");
        return 0;
    } elsif (grep {$direction eq $_} @kingside_terms) {
        _update_move_record("castle", "kingside");
        return 0;
    }
    return 1;
}
#TODO Handle castling, en passant
sub process_keyword {
    ($idx, $keywords, $coords, $pieces, $piece_info, $move_record) = @_;
    my $keyword = $keywords->{$idx};

    # NOTE As keywords expand, it can be converted to a hash lookup where each
    # keyword maps to a "platonic" keyword and its corresponding function

    my $ret = 0;
    given ($keyword) {
        when (qw(ON))                           { $ret = _on(); }
        when (qw(TO))                           { $ret = _to(); }
        when (qw(FROM))                         { $ret = _from(); }
        when ([qw(CAPTURES TAKES)])             { $ret = _capture(); }
        when (qw(CHECK))                        { $ret = _check(); }
        when ([qw(MATE CHECKMATE CHECKMATES)])  { $ret = _mate(); }
        when (qw(CASTLES))                      { $ret = _castle(); }
        when (qw(PROMOTES))                     { $ret = _promote(); }
        default                                 { return 1; }
    }

    return $ret;
}

1;
