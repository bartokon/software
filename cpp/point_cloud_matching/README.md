# Point Cloud Matching

Header-only C++20 library for point cloud processing and spatial matching.

## Purpose

Implements spatial data structures and algorithms for:
- 3D point representation
- Point cloud containers
- Nearest neighbor search
- Point matching algorithms
- Spatial queries

## Features

- Template-based design for flexibility (float/double)
- Header-only library (easy integration)
- KD-tree and brute force search
- Modern C++20 features
- Zero external dependencies

## Files

- `src/Point_3D.hpp` - 3D point class
- `src/Point_Cloud.hpp` - Point cloud container (template)
- `src/BFTree.hpp` - Brute force / KD-tree search
- `src/Point_Matcher.hpp` - Point matching algorithms
- `src/main.cpp` - Example usage and tests
- `Makefile` - Build system

## Building

```bash
make
./main.elf
```

## Requirements

- C++20 compiler (GCC 11+, Clang 12+)
- GNU Make

## Usage Example

### Basic Point Cloud

```cpp
#include "Point_Cloud.hpp"
#include "Point_3D.hpp"

// Create point cloud
Point_Cloud<double> cloud;

// Add points
cloud.add_point(Point_3D<double>(1.0, 2.0, 3.0));
cloud.add_point(Point_3D<double>(4.0, 5.0, 6.0));

// Access points
std::cout << "Size: " << cloud.size() << std::endl;
auto point = cloud.get_point(0);
```

### Nearest Neighbor Search

```cpp
#include "BFTree.hpp"

// Create search structure
BFTree<double> tree(cloud);

// Find nearest neighbor
Point_3D<double> query(2.0, 3.0, 4.0);
auto nearest = tree.find_nearest(query);
double distance = tree.distance(query, nearest);
```

### Point Matching

```cpp
#include "Point_Matcher.hpp"

// Match two point clouds
Point_Cloud<double> source, target;
// ... fill with points ...

Point_Matcher matcher;
auto correspondences = matcher.match(source, target);

// Process matches
for (const auto& match : correspondences) {
    std::cout << "Source " << match.source_idx
              << " -> Target " << match.target_idx
              << " (distance: " << match.distance << ")"
              << std::endl;
}
```

## Data Structures

### Point_3D<T>
Represents a 3D point with template type T (float/double).

**Operations:**
- Distance calculation
- Vector operations
- Transformations
- Comparison operators

### Point_Cloud<T>
Container for managing collections of points.

**Features:**
- Dynamic resizing
- Random access
- Iterators
- Statistics (centroid, bounds)

### BFTree<T>
Spatial search structure supporting:
- Brute force O(n) search
- KD-tree O(log n) search (planned)
- Radius search
- K-nearest neighbors

### Point_Matcher
Matches corresponding points between clouds.

**Algorithms:**
- Nearest neighbor matching
- Mutual matching
- Distance thresholding

## API Reference

### Point_3D<T>

```cpp
template<typename T>
class Point_3D {
public:
    T x, y, z;

    Point_3D(T x, T y, T z);

    // Distance
    T distance(const Point_3D& other) const;
    T distance_squared(const Point_3D& other) const;

    // Vector operations
    Point_3D operator+(const Point_3D& other) const;
    Point_3D operator-(const Point_3D& other) const;
    Point_3D operator*(T scalar) const;

    // Dot/cross products
    T dot(const Point_3D& other) const;
    Point_3D cross(const Point_3D& other) const;
};
```

### Point_Cloud<T>

```cpp
template<typename T>
class Point_Cloud {
public:
    void add_point(const Point_3D<T>& point);
    Point_3D<T> get_point(size_t index) const;
    size_t size() const;

    // Statistics
    Point_3D<T> centroid() const;
    void get_bounds(Point_3D<T>& min, Point_3D<T>& max) const;

    // Transformations
    void transform(const Matrix4x4& matrix);
    void translate(const Point_3D<T>& offset);
};
```

## Compilation Flags

The Makefile uses:
- `-std=c++20` - C++20 standard
- `-Wall -Wextra -Wpedantic` - All warnings
- `-O0 -g` - Debug build (change to `-O2` for release)

## Performance

**Brute Force Search:**
- Time: O(n) per query
- Space: O(n)
- Best for small clouds (<1000 points)

**KD-Tree Search (planned):**
- Time: O(log n) per query average
- Space: O(n)
- Best for large clouds (>1000 points)

## Integration

Header-only design makes integration easy:

1. Copy header files to your project
2. Include in your code
3. Compile with C++20

```cpp
#include "Point_Cloud.hpp"
#include "BFTree.hpp"
// Ready to use!
```

## Testing

The `main.cpp` file contains tests:
- Point operations
- Cloud creation
- Nearest neighbor search
- Matching algorithms

Run tests:
```bash
make
./main.elf
```

## Cleaning

```bash
make clean
```

## Learning Objectives

- Advanced C++ templates
- Header-only library design
- Spatial data structures
- Point cloud algorithms
- Modern C++20 features

## Future Enhancements

- [ ] KD-tree implementation
- [ ] Octree support
- [ ] ICP algorithm integration
- [ ] Normal estimation
- [ ] Downsampling methods

## References

- [Point Cloud Library (PCL)](https://pointclouds.org/)
- [KD-Tree](https://en.wikipedia.org/wiki/K-d_tree)
- [Nearest Neighbor Search](https://en.wikipedia.org/wiki/Nearest_neighbor_search)
