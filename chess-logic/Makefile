CC 			= gcc
CFLAGS 		= -Wall -Wextra -std=c11 -I. -Itest
DBG_FLAG	= -g

TARGET 		= chess
TARGET_DBG 	= chess_debug
SRCS 		= game.c manipulate_board.c move_validation.c render.c
OBJS 		= $(SRCS:.c=.o)
OBJS_DBG	= $(SRCS:.c=.dbg.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) 
	chmod +x $(TARGET)

debug: $(TARGET_DBG)

$(TARGET_DBG): $(OBJS_DBG)
	$(CC) $(CFLAGS) $(DBG_FLAG) -o $(TARGET_DBG) $(OBJS_DBG)
	chmod +x $(TARGET_DBG)

%.dbg.o: %.c
	$(CC) $(CFLAGS) $(DBG_FLAG) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS) $(TARGET_DBG) $(OBJS_DBG)