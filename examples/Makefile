CC = gcc
CFLAGS = -std=c99 -pedantic -Wall -Wshadow -Wextra
CFLAGS += -Wno-unused-function

LIB = ../vnn.h
SRC = $(wildcard *.c)
BIN = $(patsubst %.c, %, $(SRC))

.PHONY = all run clean

all: $(BIN) $(LIB)

run: all
	$(foreach bin,$(BIN),./$(bin))

%: %.c $(LIB)
	$(CC) $(CFLAGS) -I $(shell dirname $(LIB)) $< -o $@

clean:
	-rm $(BIN)
