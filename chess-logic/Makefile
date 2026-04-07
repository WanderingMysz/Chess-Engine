CC 		= gcc
TARGET 	= chess
CFLAGS 	= -Wall -Wextra -std=c11

SRCS 	= game.c manipulate_board.c move_validation.c render.c
OBJS 	= $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) 

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)