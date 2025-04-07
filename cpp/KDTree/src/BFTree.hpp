#ifndef BFTREE_HPP
# define BFTREE_HPP
# include <vector>
# include <queue>
# include <cmath>
# include <utility>
# include <future>
# include <Point_3D.hpp>

template <class T>
struct Point_with_neighbors {
    Point_3D<T> point;
    std::vector<std::pair<Point_3D<T>, double>> neighbors;
};

template <class T>
class BFTree {
public:
    std::vector<Point_3D<T>> points;
    std::vector<struct Point_with_neighbors<T>> points_with_neighbors;
    const size_t NEIGHBORS;

    // Constructor
    BFTree(std::vector<Point_3D<T>> points, size_t Neighbors = 4) :
        points(points), NEIGHBORS(Neighbors) {}

    // Destructor
    ~BFTree() {}

    void build_tree() {
        points_with_neighbors.reserve(points.size());
        std::vector<std::future<struct Point_with_neighbors<T>>> futures;
        size_t const MAX_THREADS = std::thread::hardware_concurrency();
        futures.reserve(MAX_THREADS);
        size_t active_threads = 0;

        printf("Building tree with %zu threads...\n", MAX_THREADS);
        for (size_t i = 0; i < points.size(); ++i) {
            if (active_threads >= MAX_THREADS) {
                for (auto &future : futures) {
                    points_with_neighbors.push_back(future.get());
                }
                futures.clear();
                active_threads = 0;
            }
            futures.emplace_back(std::async(
                std::launch::async,
                [this, i]() {
                    return this->search_nearest_neighbors(points[i]);
                }
            ));
            ++active_threads;
        }

        for (auto &future : futures) {
            points_with_neighbors.push_back(future.get());
        }
    }

    struct Point_with_neighbors<T> search_nearest_neighbors(
        Point_3D<T> const &query_point
    ) {
        struct Point_with_neighbors<T> pwn{
            query_point,
            std::vector<std::pair<Point_3D<T>, double>>()
        };
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
        for (Point_3D<T> const &p : points) {
            if (p == query_point) continue; // Skip the point itself
            double const distance = this->point_to_point_distance(
                query_point,
                p
            );
            if (neighbors_queue.size() < this->NEIGHBORS) {
                neighbors_queue.push(std::make_pair(p, distance));
            } else if (distance < neighbors_queue.top().second) {
                neighbors_queue.pop();
                neighbors_queue.push(std::make_pair(p, distance));
            }
        }

        // Transfer the points from the priority queue to the vector
        while (!neighbors_queue.empty()) {
            pwn.neighbors.push_back(neighbors_queue.top());
            neighbors_queue.pop();
        }
        return pwn;
    }

    double point_to_point_distance(
        Point_3D<T> const &p1,
        Point_3D<T> const &p2
    ) {
        double distance = std::pow(p1.x - p2.x, 2);
        distance += std::pow(p1.y - p2.y, 2);
        distance += std::pow(p1.z - p2.z, 2);
        distance = std::sqrt(distance);
        return distance;
    }

    void print() {
        for (
                struct Point_with_neighbors<T> const &pwn :
                this->points_with_neighbors
            ) {
            printf("/*******************/\n");
            pwn.point.print();
            for (std::pair<Point_3D<T>, double> const &n : pwn.neighbors) {
                printf(
                    "%s - Distance: %.3f\n",
                    n.first.print_string().c_str(),
                    n.second
                );
            }
        }
    }

};

#endif
