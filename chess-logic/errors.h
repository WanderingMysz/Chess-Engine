#ifndef ERRORS.H
#define ERRORS.H

typedef enum {
    ERR_NONE_FOUND = -1,
    ERR_MULTIPLE_FOUND = -2,
    ERR_NOTATION = -3,
    ERR_OTHER = -4
} ValidationErrorCode;

#endif