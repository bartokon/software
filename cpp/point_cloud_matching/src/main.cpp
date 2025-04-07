#include <iostream>
#include <random>

#include <Point_Cloud.hpp>
#include <BFTree.hpp>
#include <Point_Matcher.hpp>

//TODO: Add check for different neighbors

int main (void) {
    printf("Hello World!\n\n");
    srand(time(0)); // Seed for random number generation

    constexpr size_t NUM_POINTS = 2e3;
    constexpr size_t NEIGHBORS = 8;
    constexpr double MANHATTAN_DISTANCE_THRESHOLD = 0.2;
    constexpr double ANGLE_DEGREES_X = 180.0;
    constexpr double ANGLE_DEGREES_Y = 0.0;
    constexpr double ANGLE_DEGREES_Z = 0.0;
    constexpr double NOISE_LEVEL = 0.1;

    Point_Cloud<double> point_cloud_0;
    point_cloud_0.generate_random_points(NUM_POINTS);

    BFTree<double> BF_tree_0(point_cloud_0.get_points(), NEIGHBORS);

    printf("Building tree 0...\n");
    BF_tree_0.build_tree();
    printf("Tree 0 built.\n");

    Point_Cloud<double> point_cloud_1;
    point_cloud_1.set_points(point_cloud_0.get_points());
    point_cloud_1.rotate_x(ANGLE_DEGREES_X);
    point_cloud_1.rotate_y(ANGLE_DEGREES_Y);
    point_cloud_1.rotate_z(ANGLE_DEGREES_Z);
    point_cloud_1.add_noise(NOISE_LEVEL);

    BFTree<double> BF_tree_1(point_cloud_1.get_points(), NEIGHBORS);

    printf("Building tree 1...\n");
    BF_tree_1.build_tree();
    printf("Tree 1 built.\n");

    Point_Matcher<double> point_matcher(
        BF_tree_0.points_with_neighbors,
        BF_tree_1.points_with_neighbors
    );

    printf("Matching points...\n");
    point_matcher.similarity_mp(MANHATTAN_DISTANCE_THRESHOLD);
    printf("Points matched.\n");
    point_matcher.print_matched_points();
    printf("Finished.\n");
    return 0;
}
