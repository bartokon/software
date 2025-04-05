# Compiler to use
CC = g++-11

# Compilation flags
CFLAGS = -std=c++20 -Wall -Wextra -O2 -g

SRC_DIR = src
OBJ_DIR = obj

# Include directories (adjust as needed)
INCLUDES = -I$(SRC_DIR)

# Source files (adjust as needed)
CPP_SRCS = $(wildcard $(SRC_DIR)/*.cpp)
HPP_SRCS = $(wildcard $(SRC_DIR)/*.hpp)

# Object files (automatically generated from source files)
OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(CPP_SRCS))
OBJS += $(patsubst $(SRC_DIR)/%.hpp, $(OBJ_DIR)/%.o, $(HPP_SRCS))

# Name of the final executable
TARGET = main.elf

print:
	@echo $(CPP_SRCS)
	@echo $(HPP_SRCS)
	@echo $(OBJS)

# Default rule: compile and link
all: $(TARGET)

# Rule to compile object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Compiling $< to $@"
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Rule to compile object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.hpp
	@echo "Compiling $< to $@"
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Rule to link object files into the executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET)

# Clean rule: remove object files and the executable
clean:
#	rm -f $(OBJS)
	rm -f $(TARGET)

# Optional: run the executable
run: $(TARGET)
	./$(TARGET)
