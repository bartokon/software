#include <stdio.h>

#include "vadd.cu"

void init_array(int *array, unsigned int size) {
    for (unsigned int i = 0; i < size; ++i) {
        array[i] = i;
    }
};

int check_array(int *array, unsigned int size) {
    for (unsigned int i = 0; i < size; ++i) {
        if (array[i] != i * 2) {
            printf("Kernel failed.");
            return 1;
        };
    }
    printf("All good.");
    return 0;
};

int main(void){
    static int a[1024];
    static int b[1024];
    static int c[1024];
    init_array(a, 1024);
    init_array(b, 1024);

    vadd_wrapper(a, b, c, 1024);

    return check_array(c, 1024);
}