.PHONY: CLI CLI-clean

CC 			= gcc
CFLAGS 		= -Wall -Wextra -std=c11 -I$(INC_DIR)
# DBG_FLAG	= -g

SRC_DIR		= $(PROJECT_ROOT)/chess-logic/src
INC_DIR		= $(PROJECT_ROOT)/chess-logic/include
OBJ_DIR		= $(PROJECT_ROOT)/build/obj
BIN_DIR		= $(PROJECT_ROOT)/build/bin

TARGET 		= chess
TAR_PATH 	= $(BIN_DIR)/$(TARGET)
SRCS 		= $(wildcard $(SRC_DIR)/*.c)
OBJS 		= $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))

CLI: $(TAR_PATH)

$(TAR_PATH): $(OBJS)
	$(CC) $(CFLAGS) -o $(TAR_PATH) $(OBJS) 
	chmod +x $(TAR_PATH)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

CLI-clean:
	rm -f $(TAR_PATH) $(OBJS)