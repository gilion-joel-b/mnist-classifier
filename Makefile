OBJECTS = main.cpp mnist.hpp
EXEC = model
CC = g++                                                              
FLAGS = -std=c++20 -O3 -Wall -Wextra

SRC = $(wildcard src/*.cpp)
HEADERS = $(wildcard src/*.hpp)

# Perform action on all object files (May or may not exist)           
all: $(SRC) $(HEADERS)
	$(CC) $(SRC) -o $(EXEC) $(FLAGS)

clean:                                                                  
	rm -f $(EXEC) src/$(OBJECTS)
