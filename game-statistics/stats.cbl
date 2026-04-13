       IDENTIFICATION DIVISION.
       PROGRAM-ID. STATS.
       AUTHOR. WanderingMysz.

       ENVIRONMENT DIVISION.
       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
           SELECT INPUT-FILE  ASSIGN TO "games.dat"
               ORGANIZATION IS LINE SEQUENTIAL.
           SELECT OUTPUT-FILE ASSIGN TO "output.dat"
               ORGANIZATION IS LINE SEQUENTIAL.

       DATA DIVISION.
       FILE SECTION.
       FD GAME-FILE.
       01 MOVE-RECORD.
           05 TURN_NUMBER PIC 9999.
           05 PLAYER_COLOR PIC X.
               88 WHITE VALUE 'W'.
               88 BLACK VALUE 'B'.
           05 FILLER PIC X.
           05 PIECE_TYPE PIC X.
               88 PT-KNIGHT    VALUE 'N'.
               88 PT-BISHOP    VALUE 'B'.
               88 PT-ROOK      VALUE 'R'.
               88 PT-QUEEN     VALUE 'Q'.
               88 PT-KING      VALUE 'K'.
           05 FILLER PIC X.
           05 FROM_SQUARE PIC 99.
           05 FILLER PIC XX.
           05 TO_SQUARE PIC 99.
           05 FILLER PIC X.
           05 CAPTURE PIC 9.
           05 PROMOTION PIC X.
               88 PR-KNIGHT    VALUE 'N'.
               88 PR-BISHOP    VALUE 'B'.
               88 PR-ROOK      VALUE 'R'.
               88 PR-QUEEN     VALUE 'Q'.
               88 PR-KING      VALUE 'K'.
           05 CHECK PIC 9.
           05 CHECKMATE PIC 9.

       WORKING-STORAGE SECTION.
       01 WS-EOF PIC X VALUE 'N'.

       PROCEDURE DIVISION.
       MAIN-PARAGRAPH.
           OPEN INPUT  INPUT-FILE
           OPEN OUTPUT OUTPUT-FILE
           PERFORM UNTIL WS-EOF = 'Y'
               READ INPUT-FILE
                   AT END MOVE 'Y' TO WS-EOF
                   NOT AT END PERFORM PROCESS-RECORD
               END-READ
           END-PERFORM
           CLOSE INPUT-FILE
           CLOSE OUTPUT-FILE
           STOP RUN.

       PROCESS-RECORD.
           CONTINUE.