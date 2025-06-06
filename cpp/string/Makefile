# Compiler to use
CC = g++-11

# Compilation flags
CFLAGS = -std=c++20 -Wall -Wextra -Wpedantic -O0 -g
#CFLAGS = -Wall -Wextra -O2 -g

SRC_DIR = src
OBJ_DIR = obj

# Include directories (adjust as needed)
INCLUDES = -I$(SRC_DIR)

# Source files (adjust as needed)
CPP_SRCS = $(wildcard $(SRC_DIR)/*.cpp)

# Object files (automatically generated from source files)
OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(CPP_SRCS))

HPP_SRCS = $(wildcard $(SRC_DIR)/*.hpp)

# Name of the final executable
TARGET = main.elf

print:
	@echo $(CPP_SRCS)
	@echo $(HPP_SRCS)
	@echo $(OBJS)

# Default rule: compile and link
all: $(TARGET)

# Rule to compile object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(HPP_SRCS)
	@echo "Compiling $< to $@"
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Rule to link object files into the executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET)

# Clean rule: remove object files and the executable
clean:
	rm -f $(OBJS)
	rm -f $(TARGET)

# Rule to create the object directory if it doesn't exist
$(OBJ_DIR):
	@if [ ! -d "$(OBJ_DIR)" ]; then mkdir -p $(OBJ_DIR); fi

# Optional: run the executable
run: obj $(TARGET)
	./$(TARGET)
