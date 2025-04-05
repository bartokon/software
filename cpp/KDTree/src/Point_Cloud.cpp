#ifndef POINT_CLOUD_HPP
#define POINT_CLOUD_HPP

#include <vector>
#include <random>

#include <Point_3D.cpp>

template <class T>
class Point_Cloud {
public:
    std::vector<Point_3D<T>> points;
    // Constructor
    Point_Cloud() {}

    // Destructor
    ~Point_Cloud() {}

    // Method to add a point to the cloud
    void add_point(Point_3D<T> point) {
        points.push_back(point);
    }

    // Method to clear the cloud
    void clear() {
        points.clear();
    }

    // Generate random points in the cloud
    void generate_random_points(size_t const num_points) {
        for (size_t i = 0; i < num_points; ++i) {
            T const x = static_cast<T>(rand()) / RAND_MAX;
            T const y = static_cast<T>(rand()) / RAND_MAX;
            T const z = static_cast<T>(rand()) / RAND_MAX;
            this->add_point(Point_3D<T>(x, y, z));
        }
    }

    // Method to print all points in the cloud
    void print() const {
        for (Point_3D<T> const &p : points) {
            p.print();
        }
    }

    // Method to get the points in the cloud
    std::vector<Point_3D<T>> get_points(){
        return this->points;
    }

};

#endif