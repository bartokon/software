# Custom String Class Implementation

Educational implementation of a std::string-like class from scratch.

## Purpose

Learning project to understand:
- Dynamic memory management
- Copy/move semantics (Rule of Five)
- Operator overloading
- STL-like interface design
- Memory optimization techniques

## Files

- `src/string.hpp` - Custom string class (4,209 lines!)
- `src/main.cpp` - Comparison with std::string
- `Makefile` - Build system

## Building

```bash
make
./string_test.elf
```

## Requirements

- C++20 compiler (GCC 11+)
- GNU Make

## Why 4,200 Lines?

This is a comprehensive learning implementation that includes:
- All basic string operations
- STL-like interface
- Memory management
- Iterator support
- Extensive operator overloading
- Comparison functions
- Search algorithms
- Conversion functions

The goal is educational - implementing features to understand how std::string works internally.

## Features Implemented

### Memory Management
- Dynamic allocation/deallocation
- Capacity management
- Small string optimization (SSO)
- Copy-on-write (optional)

### Constructors
```cpp
MyString s1;                    // Default
MyString s2("Hello");           // C-string
MyString s3(s2);                // Copy
MyString s4(std::move(s2));     // Move
MyString s5(10, 'a');           // Fill
```

### Operators
```cpp
s1 + s2          // Concatenation
s1 += s2         // Append
s1 == s2         // Comparison
s1 < s2          // Lexicographic
s1[i]            // Indexing
```

### Methods
```cpp
s.size()         // Length
s.empty()        // Is empty
s.clear()        // Clear content
s.append(...)    // Append
s.substr(...)    // Substring
s.find(...)      // Search
s.replace(...)   // Replace
```

### Iterators
```cpp
for (char c : myString) {
    std::cout << c;
}
```

## Usage Example

```cpp
#include "string.hpp"

int main() {
    // Create strings
    MyString s1 = "Hello";
    MyString s2 = "World";

    // Concatenate
    MyString s3 = s1 + " " + s2;  // "Hello World"

    // Search
    size_t pos = s3.find("World");  // Returns 6

    // Substring
    MyString sub = s3.substr(0, 5);  // "Hello"

    // Comparison
    if (s1 < s2) {
        std::cout << s1 << " comes before " << s2 << std::endl;
    }

    // Iteration
    for (char c : s3) {
        std::cout << c << std::endl;
    }

    return 0;
}
```

## Comparison with std::string

The `main.cpp` demonstrates that MyString behaves like std::string:

```cpp
// Custom string
MyString ms = "Test";
ms += " String";
std::cout << ms.size() << std::endl;

// std::string
std::string ss = "Test";
ss += " String";
std::cout << ss.size() << std::endl;

// Should behave identically
```

## Rule of Five

The class properly implements:

```cpp
class MyString {
public:
    // 1. Destructor
    ~MyString();

    // 2. Copy constructor
    MyString(const MyString& other);

    // 3. Copy assignment
    MyString& operator=(const MyString& other);

    // 4. Move constructor
    MyString(MyString&& other) noexcept;

    // 5. Move assignment
    MyString& operator=(MyString&& other) noexcept;
};
```

## Memory Management

### Allocation Strategy
```cpp
// Start with small capacity
char* data = new char[capacity + 1];

// Grow when needed (typically 2x growth)
if (new_size > capacity) {
    resize(capacity * 2);
}
```

### Small String Optimization (SSO)
For strings ≤ 15 chars, store in object directly:
```cpp
union {
    char* heap_ptr;      // For large strings
    char buffer[16];     // For small strings
};
```

## Performance Considerations

**Time Complexity:**
- Access: O(1)
- Append: O(1) amortized
- Insert: O(n)
- Find: O(n*m)
- Compare: O(n)

**Memory:**
- Overhead: ~24-32 bytes per string
- SSO eliminates allocation for small strings

## Testing

The main program tests:
- Construction and destruction
- Copy and move semantics
- String operations
- Memory management
- Comparison with std::string

Run tests:
```bash
make
./string_test.elf
```

Expected output shows MyString behaving identically to std::string.

## Compilation Flags

```bash
g++ -std=c++20 -Wall -Wextra -Wpedantic -O0 -g -o string_test.elf src/main.cpp
```

- `-std=c++20` - C++20 features
- `-Wall -Wextra -Wpedantic` - All warnings
- `-O0 -g` - Debug mode (change to `-O2` for performance)

## Cleaning

```bash
make clean
```

## Learning Objectives

- Deep understanding of std::string internals
- Memory management in C++
- RAII (Resource Acquisition Is Initialization)
- Rule of Five/Zero
- Operator overloading patterns
- Iterator implementation
- Exception safety
- Performance optimization

## Common Pitfalls (Avoided)

1. **Memory leaks**: Properly delete[] in destructor
2. **Double free**: Move constructor sets source pointer to nullptr
3. **Self-assignment**: Check for `this != &other`
4. **Buffer overrun**: Always null-terminate strings
5. **Iterator invalidation**: Clear iterators on reallocation

## Comparison to Production Code

**This Implementation:**
- Educational focus
- Readable over optimized
- All features in one file

**std::string:**
- Highly optimized
- Platform-specific
- Extensive testing
- Edge case handling

**For production code, always use std::string!**

## References

- [cppreference: std::string](https://en.cppreference.com/w/cpp/string/basic_string)
- [The Rule of Five](https://en.cppreference.com/w/cpp/language/rule_of_three)
- [RAII](https://en.cppreference.com/w/cpp/language/raii)
- [Small String Optimization](https://blogs.msmvps.com/gdicanio/2016/11/17/the-small-string-optimization/)
