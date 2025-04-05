#include <iostream>

#include <Point_Cloud.cpp>
#include <BFTree.cpp>

int main (void) {
    printf("Hello World!\n");
    constexpr size_t NUM_POINTS = 4;

    Point_Cloud<float> point_cloud;
    point_cloud.generate_random_points(NUM_POINTS);
    point_cloud.print();
    BFTree<float> BF_tree(point_cloud.get_points());
    BF_tree.build_tree();
    BF_tree.print();
    return 0;
}
