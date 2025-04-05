#ifndef POINT_3D_HPP
#define POINT_3D_HPP

#include <iostream>

template <class T>
class Point_3D {
public:
    T x, y, z;
    // Constructor
    Point_3D(T x, T y, T z) : x(x), y(y), z(z) {}

    // Destructor
    ~Point_3D() {}

    // Method to print the point
    void print() const {
        printf("Point(%.3f, %.3f, %.3f)\n", x, y, z);
    }

    bool operator==(Point_3D<T> const &other) const {
        return (this->x == other.x && this->y == other.y && this->z == other.z);
    }
};

#endif