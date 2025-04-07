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
};

template <class T>
class Point_Matcher {
public:
    std::vector<struct Point_with_neighbors<T>> points_with_neighbors_0;
    std::vector<struct Point_with_neighbors<T>> points_with_neighbors_1;
    std::vector<struct Matched_Points<T>> matched_points;

    Point_Matcher(
        std::vector<struct Point_with_neighbors<T>> points_with_neighbors_0,
        std::vector<struct Point_with_neighbors<T>> points_with_neighbors_1
    ) : points_with_neighbors_0(points_with_neighbors_0),
        points_with_neighbors_1(points_with_neighbors_1) {}

    void cosine_similarity(double threshold = 0.5) {
        for (size_t i = 0; i < points_with_neighbors_0.size(); ++i) {
            struct Matched_Points<T> matched_point(points_with_neighbors_0[i], points_with_neighbors_1[0], -1.0);
            matched_point.point_0 = points_with_neighbors_0[i];
            for (size_t j = 0; j < points_with_neighbors_1.size(); ++j) {
                double dot_product_value = this->dot_product(
                    points_with_neighbors_0[i].neighbors,
                    points_with_neighbors_1[j].neighbors
                );
                double magnitude_0 = this->magnitude(points_with_neighbors_0[i].neighbors);
                double magnitude_1 = this->magnitude(points_with_neighbors_1[j].neighbors);
                double cosine_similarity = -1.0;
                if (magnitude_0 == 0 || magnitude_1 == 0) {
                    printf("One of the vectors is zero, cosine similarity is undefined.\n");
                } else {
                    cosine_similarity = dot_product_value / (magnitude_0 * magnitude_1);
                    //printf("Cosine similarity between point %zu and point %zu: %.3f\n", i, j, cosine_similarity);
                }
                if (cosine_similarity > matched_point.cosine_similarity) {
                    matched_point.point_1 = points_with_neighbors_1[j];
                    matched_point.cosine_similarity = cosine_similarity;
                }
                if (cosine_similarity >= 1.0) {
                    //printf("Dot product: %.3f, Magnitude 0: %.3f, Magnitude 1: %.3f\n", dot_product_value, magnitude_0, magnitude_1);
                    break; // Early exit if perfect match is found
                }
            }
            if (matched_point.cosine_similarity > threshold) {
                if (matched_point.cosine_similarity < 0.999) {
                    printf("Matched points with cosine similarity above threshold: %.3f\n", matched_point.cosine_similarity);
                }
                matched_points.push_back(matched_point);
            }
        }
    }

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

    double magnitude(
        std::vector<std::pair<Point_3D<T>, double>> neighbors
    ) {
        std::vector<double> v;
        for (std::pair<Point_3D<T>, double> const &n : neighbors) {
            v.push_back(n.second);
        }
        return std::sqrt(std::accumulate(v.begin(), v.end(), 0.0, [](double a, double b) { return a + b * b; }));
    }

    ~Point_Matcher() {}
};

#endif
