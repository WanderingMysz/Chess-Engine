#include "move_validation.h"

bool is_valid_move(char* move) {
    const char* pattern = "^(O-O(-O)?|[NBRQK]?[a-h]?x?[a-h][1-8])[+#]?$";

    regex_t re;
    if (regcomp(&re, pattern, REG_EXTENDED | REG_ICASE)){
        printf("Failed to compile regex\n");
        return false;
    }

    int status = regexec(&re, move, 0, NULL, 0);
    regfree(&re);

    return status == 0;
}