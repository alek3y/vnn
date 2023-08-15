CC = gcc
CFLAGS = -std=c99 -g -pedantic -Wall -Wshadow -Wextra
CFLAGS += -Wno-unused-function
LDFLAGS = -lm

LIB = ../vnn.h
SRC = $(wildcard *.c)
BIN = $(patsubst %.c, %, $(SRC))

.PHONY = all run clean

all: $(BIN) $(LIB)

run: all
	@$(foreach bin,$(BIN),echo "=== ./$(bin) ==="; ./$(bin);)

%: %.c $(LIB)
	$(CC) $(CFLAGS) $(LDFLAGS) -I $(shell dirname $(LIB)) $< -o $@

clean:
	-rm $(BIN)
