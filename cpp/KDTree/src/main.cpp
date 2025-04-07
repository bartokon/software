#include <iostream>
#include <random>

#include <Point_Cloud.hpp>
#include <BFTree.hpp>
#include <PointMatcher.hpp>

int main (void) {
    printf("Hello World!\n\n");
    srand(time(0)); // Seed for random number generation

    constexpr size_t NUM_POINTS = 16;
    constexpr size_t NEIGHBORS = 4;
    constexpr double MANHATTAN_DISTANCE_THRESHOLD = 0.2;

    Point_Cloud<float> point_cloud_0;
    point_cloud_0.generate_random_points(NUM_POINTS);
    BFTree<float, NEIGHBORS> BF_tree_0(point_cloud_0.get_points());
    BF_tree_0.build_tree();

    Point_Cloud<float> point_cloud_1;
    point_cloud_1.generate_random_points(NUM_POINTS);
    BFTree<float, NEIGHBORS> BF_tree_1(point_cloud_1.get_points());
    BF_tree_1.build_tree();

    Point_Matcher<float> point_matcher(
        BF_tree_0.points_with_neighbors,
        BF_tree_1.points_with_neighbors
    );

    point_matcher.similarity(MANHATTAN_DISTANCE_THRESHOLD);
    point_matcher.print_matched_points();
    printf("Point Matcher finished.\n");
    return 0;
}
