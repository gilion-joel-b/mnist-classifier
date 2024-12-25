OBJECTS = main.cpp
EXEC = model
CC = g++                                                              

# Perform action on all object files (May or may not exist)           
all: src/$(OBJECTS)                                                       
	$(CC)  -o $(EXEC) src/$(OBJECTS)
