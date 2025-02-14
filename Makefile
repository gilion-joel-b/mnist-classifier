OBJECTS = main.cpp
EXEC = model
CC = g++                                                              
FLAGS = -std=c++20 -O2 -Wall -Wextra

# Perform action on all object files (May or may not exist)           
all: src/$(OBJECTS)                                                       
	$(CC)  -o $(EXEC) src/$(OBJECTS) $(FLAGS)
