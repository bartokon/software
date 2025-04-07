#ifndef POINTMATCHER_HPP
# define POINTMATCHER_HPP
# include <vector>
# include <numeric>
# include <BFTree.hpp>

template <class T>
struct Matched_Points {
    struct Point_with_neighbors<T> point_0;
    struct Point_with_neighbors<T> point_1;
    double cosine_similarity;
    double manhattan_distance;
};

template <class T>
class Point_Matcher {
public:
    std::vector<struct Point_with_neighbors<T>> points_with_neighbors_0;
    std::vector<struct Point_with_neighbors<T>> points_with_neighbors_1;
    std::vector<struct Matched_Points<T>> matched_points;

    // Constructor
    Point_Matcher(
        std::vector<struct Point_with_neighbors<T>> points_with_neighbors_0,
        std::vector<struct Point_with_neighbors<T>> points_with_neighbors_1
    ) : points_with_neighbors_0(points_with_neighbors_0),
        points_with_neighbors_1(points_with_neighbors_1) {}

    // Destructor
    ~Point_Matcher() {}

    double cosine_similarity(
        struct Point_with_neighbors<T> &point_0,
        struct Point_with_neighbors<T> &point_1
    ) {
        double dot_product_value = this->dot_product(
            point_0.neighbors,
            point_1.neighbors
        );
        double magnitude_0 = this->magnitude(
            point_0.neighbors
        );
        double magnitude_1 = this->magnitude(
            point_1.neighbors
        );
        double cosine_similarity = -1.0;
        if (magnitude_0 == 0 || magnitude_1 == 0) {
            printf(
                "One of the vectors is zero, cosine similarity is"
                "undefined.\n"
            );
        } else {
            cosine_similarity = dot_product_value;
            cosine_similarity /= (magnitude_0 * magnitude_1);
        }
        return cosine_similarity;
    }

    // Method to calculate Manhattan distance
    double manhattan_distance(
        struct Point_with_neighbors<T> &point_0,
        struct Point_with_neighbors<T> &point_1
    ) {
        double distance = 0.0;
        for (size_t i = 0; i < point_0.neighbors.size(); ++i) {
            distance += std::abs(
                point_0.neighbors[i].second - point_1.neighbors[i].second
            );
        }
        return distance;
    }

    // Method to match points based on cosine similarity
    void similarity(
        double manhattan_distance_threshold = 0.1
    ) {
        for (size_t i = 0; i < points_with_neighbors_0.size(); ++i) {
            struct Matched_Points<T> matched_point(
                points_with_neighbors_0[i],
                points_with_neighbors_1[0],
                -1.0,
                std::numeric_limits<double>::max()
            );
            matched_point.point_0 = points_with_neighbors_0[i];
            for (size_t j = 0; j < points_with_neighbors_1.size(); ++j) {
                double const cs = this->cosine_similarity(
                    points_with_neighbors_0[i],
                    points_with_neighbors_1[j]
                );
                double const md = this->manhattan_distance(
                    points_with_neighbors_0[i],
                    points_with_neighbors_1[j]
                );
                if (md < matched_point.manhattan_distance) {
                    matched_point.manhattan_distance = md;
                    matched_point.cosine_similarity = cs;
                    matched_point.point_1 = points_with_neighbors_1[j];
                }
                //Perfect Manhattan distance
                if (matched_point.manhattan_distance == 0.0) {
                    break;
                }
            }
            if (matched_point.manhattan_distance < manhattan_distance_threshold) {
                matched_points.push_back(matched_point);
            }
        }
    }

    // Calculate dot product
    double dot_product(
        std::vector<std::pair<Point_3D<T>, double>> neighbors_0,
        std::vector<std::pair<Point_3D<T>, double>> neighbors_1
    ) {
        std::vector<double> v0;
        std::vector<double> v1;
        for (std::pair<Point_3D<T>, double> const &n : neighbors_0) {
            v0.push_back(n.second);
        }
        for (std::pair<Point_3D<T>, double> const &n : neighbors_1) {
            v1.push_back(n.second);
        }
        return std::inner_product(v0.begin(), v0.end(), v1.begin(), 0.0);
    }

    // Calculate magnitude
    double magnitude(
        std::vector<std::pair<Point_3D<T>, double>> neighbors
    ) {
        std::vector<double> v;
        for (std::pair<Point_3D<T>, double> const &n : neighbors) {
            v.push_back(n.second);
        }
        return std::sqrt(
            std::accumulate(
                v.begin(), v.end(), 0.0,
                [](double a, double b) { return a + b * b; }
            )
        );
    }

    void print_matched_points() {
        for (const auto &matched_point : matched_points) {
            printf(
                "Matched Points:\n"
                "Point 0: %s\n"
                "Point 1: %s\n"
                "Cosine Similarity: %.3f\n"
                "Manhattan Distance: %.3f\n",
                matched_point.point_0.point.print_string().c_str(),
                matched_point.point_1.point.print_string().c_str(),
                matched_point.cosine_similarity,
                matched_point.manhattan_distance
            );
            printf("Neighbors 0:\n");
            for (const auto &neighbor : matched_point.point_0.neighbors) {
                printf(
                    "  %s -> %f\n",
                    neighbor.first.print_string().c_str(),
                    neighbor.second
                );
            }
            printf("Neighbors 1:\n");
            for (const auto &neighbor : matched_point.point_1.neighbors) {
                printf(
                    "  %s -> %f\n",
                    neighbor.first.print_string().c_str(),
                    neighbor.second
                );
            }
            printf("\n");
        }
    }

};

#endif
