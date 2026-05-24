#!/usr/bin/env gawk -f

BEGIN {
    block_field;
    s_break;
    enumerate;
}
{
    for (i = 1; i <= NF; i++) {
        if (s_break) break;

        switch ($i) {
            # Skip line comments
            case /;/:
                s_break++;
                break;

            # Skip metadata [] and annotations {} (assume well-formatted .pgn)
            case /[\[\{]/:
                block_field++;
                break;
            case /[\]\}]/:
                block_field--;
                break;

            default:
                if (block_field) break;

                # If it matches a move number, e.g. "1.a4"
                if ($i ~ /^[0-9]+(\.)[^.][a-zA-z0-9]*/) {
                    if (!enumerate) sub(/^[0-9]+(\.)+/, "", $i); 
                    printf "%s", $i;
                }
                else {
                    # Remove reference to move number, e.g. "1...a5"
                    sub(/^[0-9]+(\.)+/, "", $i); 
                    printf " %s\n", $i;
                }
        }
        
    }
    s_break = 0;
}