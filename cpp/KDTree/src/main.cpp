#include <iostream>
#include <random>

#include <Point_Cloud.hpp>
#include <BFTree.hpp>
#include <PointMatcher.hpp>

//TODO: Add check for different neighbors

int main (void) {
    printf("Hello World!\n\n");
    srand(time(0)); // Seed for random number generation

    constexpr size_t NUM_POINTS = 16;
    constexpr size_t NEIGHBORS = 4;
    constexpr double MANHATTAN_DISTANCE_THRESHOLD = 0.2;
    constexpr double ANGLE_DEGREES_X = 180.0;
    constexpr double ANGLE_DEGREES_Y = 0.0;
    constexpr double ANGLE_DEGREES_Z = 0.0;

    Point_Cloud<double> point_cloud_0;
    point_cloud_0.generate_random_points(NUM_POINTS);
    BFTree<double> BF_tree_0(point_cloud_0.get_points(), NEIGHBORS);
    BF_tree_0.build_tree();

    Point_Cloud<double> point_cloud_1;
    point_cloud_1.set_points(point_cloud_0.get_points());
    point_cloud_1.rotate_x(ANGLE_DEGREES_X);
    point_cloud_1.rotate_y(ANGLE_DEGREES_Y);
    point_cloud_1.rotate_z(ANGLE_DEGREES_Z);

    BFTree<double> BF_tree_1(point_cloud_1.get_points(), NEIGHBORS);
    BF_tree_1.build_tree();

    Point_Matcher<double> point_matcher(
        BF_tree_0.points_with_neighbors,
        BF_tree_1.points_with_neighbors
    );

    point_matcher.similarity(MANHATTAN_DISTANCE_THRESHOLD);
    point_matcher.print_matched_points();
    printf("Point Matcher finished.\n");
    return 0;
}
