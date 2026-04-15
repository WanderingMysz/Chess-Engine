CC 		= gcc
CFLAGS 	= -Wall -Wextra -std=c11 -I. -Itest

TARGET 	= chess
SRCS 	= game.c manipulate_board.c move_validation.c render.c
OBJS 	= $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) 

test: $(TARGET_TEST)
	./$(TARGET_TEST)

$(TARGET_TEST): $(OBJS_TEST)
	$(CC) $(CFLAGS) -o $(TARGET_TEST) $(OBJS_TEST)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS) $(TARGET_TEST) $(OBJS_TEST)