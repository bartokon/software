#ifndef BFTREE_HPP
#define BFTREE_HPP

#include <vector>
#include <cmath>
#include <utility>
#include <queue>
#include <algorithm>
#include "Point_3D.hpp"

template <class T>
struct Point_with_neighbors {
    struct Comparator {
        bool operator()(
            std::pair<Point_3D<T>, double> const &lhs,
            std::pair<Point_3D<T>, double> const &rhs
        ) const {
            return lhs.second > rhs.second;
        }
    };

    std::priority_queue<
        std::pair<Point_3D<T>, double>,
        std::vector<std::pair<Point_3D<T>, double>>,
        Comparator
    > neighbors_queue;

    Point_with_neighbors(Point_3D<T> const &p)
    : neighbors_queue(Comparator()) {}
};

template <class T, size_t Neighbors = 4>
class BFTree {
public:
    std::vector<Point_3D<T>> points;
    std::vector<struct Point_with_neighbors<T>> points_with_neighbors;

    // Constructor
    BFTree(std::vector<Point_3D<T>> points) : points(points){}

    void build_tree() {
        for (Point_3D<T> const &p : points) {
            points_with_neighbors.push_back(this->search_nearest_neighbors(p));
        }
    }

    struct Point_with_neighbors<T> search_nearest_neighbors(Point_3D<T> const &query_point) {
        struct Point_with_neighbors<T> pwn{query_point};
        for (Point_3D<T> const &p : points) {
            if (p == query_point) continue; // Skip the point itself
            double const distance = this->point_to_point_distance(query_point, p);
            if (pwn.neighbors_queue.size() < Neighbors) {
                pwn.neighbors_queue.push(std::make_pair(p, distance));
            } else if (distance < pwn.neighbors_queue.top().second) {
                pwn.neighbors_queue.pop();
                pwn.neighbors_queue.push(std::make_pair(p, distance));
            }
        }
        return pwn;
    }

    double point_to_point_distance(Point_3D<T> const &p1, Point_3D<T> const &p2) {
        double distance = std::pow(p1.x - p2.x, 2);
        distance += std::pow(p1.y - p2.y, 2);
        distance += std::pow(p1.z - p2.z, 2);
        distance = std::sqrt(distance);
        return distance;
    }

    void print() {
        for (struct Point_with_neighbors<T> const &pwn : points_with_neighbors) {
            for (std::pair<Point_3D<T>, double> const &n : pwn.neighbors_queue) {
                printf("Point(%.3f, %.3f, %.3f) - Distance: %.3f\n",
                    n.first.x, n.first.y, n.first.z,
                    n.second
                );
            }
        }
    }

    // Destructor
    ~BFTree() {}

};

#endif
