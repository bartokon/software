#ifndef POINT_CLOUD_HPP
# define POINT_CLOUD_HPP
# include <vector>
# include <random>
# include <cmath>
# include <Point_3D.hpp>

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
        points.reserve(num_points);
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

    // Method to set the points in the cloud
    void set_points(std::vector<Point_3D<T>> const &points) {
        this->points = points;
    }

    // Method to rotate the points in the cloud in X axis
    void rotate_x(double const angle_degrees) {
        double const angle_radians = angle_degrees * M_PI / 180.0;
        double const cos_theta = std::cos(angle_radians);
        double const sin_theta = std::sin(angle_radians);
        for (Point_3D<T> &p : points) {
            p.y = p.y * cos_theta - p.z * sin_theta;
            p.z = p.y * sin_theta + p.z * cos_theta;
        }
    }

    // Method to rotate the points in the cloud in Y axis
    void rotate_y(double const angle_degrees) {
        double const angle_radians = angle_degrees * M_PI / 180.0;
        double const cos_theta = std::cos(angle_radians);
        double const sin_theta = std::sin(angle_radians);
        for (Point_3D<T> &p : points) {
            p.x = p.x * cos_theta + p.z * sin_theta;
            p.z = -p.x * sin_theta + p.z * cos_theta;
        }
    }

    // Method to rotate the points in the cloud in Z axis
    void rotate_z(double const angle_degrees) {
        double const angle_radians = angle_degrees * M_PI / 180.0;
        double const cos_theta = std::cos(angle_radians);
        double const sin_theta = std::sin(angle_radians);
        for (Point_3D<T> &p : points) {
            p.x = p.x * cos_theta - p.y * sin_theta;
            p.y = p.x * sin_theta + p.y * cos_theta;
        }
    }

    void add_noise(double const noise) {
        for (Point_3D<T> &p : points) {
            p.x += static_cast<T>(rand()) / RAND_MAX * noise;
            p.y += static_cast<T>(rand()) / RAND_MAX * noise;
            p.z += static_cast<T>(rand()) / RAND_MAX * noise;
        }
    }

    void add_noise(double const noise_x, double const noise_y, double const noise_z) {
        for (Point_3D<T> &p : points) {
            p.x += static_cast<T>(rand()) / RAND_MAX * noise_x;
            p.y += static_cast<T>(rand()) / RAND_MAX * noise_y;
            p.z += static_cast<T>(rand()) / RAND_MAX * noise_z;
        }
    }
};

#endif