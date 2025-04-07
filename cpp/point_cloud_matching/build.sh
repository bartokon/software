#!/bin/bash
set -e

# Compile the C code
gcc -c src/main.c -o main.o
# Link the object file to create an executable
gcc main.o -o main.elf
# Run the executable
./main.elf
