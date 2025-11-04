# Bubble Sort with Pointers

Classic bubble sort algorithm implementation using C pointers.

## Purpose

Educational project demonstrating:
- Pointer arithmetic in C
- Array manipulation via pointers
- Basic sorting algorithms
- Memory access patterns

## Algorithm

Bubble sort compares adjacent elements and swaps them if they're in wrong order. This implementation uses pointers instead of array indices.

**Time Complexity**: O(n²)
**Space Complexity**: O(1)
**Stable**: Yes

## Files

- `src/main.c` - Implementation and test code
- `build.sh` - Build script

## Building

```bash
./build.sh
```

Or manually:
```bash
gcc src/main.c -o main
```

## Running

```bash
./main
```

Expected output:
```
Original array: [5, 2, 9, 1, 5, 6]
Sorted array:   [1, 2, 5, 5, 6, 9]
```

## Code Example

```c
void bubble_sort(int *arr, int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (*(arr + j) > *(arr + j + 1)) {
                // Swap using pointers
                int temp = *(arr + j);
                *(arr + j) = *(arr + j + 1);
                *(arr + j + 1) = temp;
            }
        }
    }
}
```

## Learning Objectives

- Understanding pointer arithmetic
- Dereferencing pointers
- Pointer-based array access
- Sorting algorithm implementation

## Notes

This is a teaching implementation. For production code, use:
- `std::sort()` (C++)
- `qsort()` (C)
- More efficient algorithms (quicksort, mergesort)

## References

- [Bubble Sort](https://en.wikipedia.org/wiki/Bubble_sort)
- C pointer basics
