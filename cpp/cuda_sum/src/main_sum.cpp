#include <stdio.h>
#include <stdlib.h>

#include "sum.hpp"

void init_array(int *array, unsigned int size) {
    for (int i = 0; i < size; ++i) {
        array[i] = i;
    }
};

int sum_array(int *array, unsigned int size) {
    int sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += array[i];
    }
    return sum;
};

int main (void) {
    const unsigned int SIZE = 1024 + 1;
    int *in = (int*)malloc(sizeof(int) * SIZE);
    int out = -1;

    init_array(in, SIZE);
    sum_wrapper(in, &out, SIZE);
    const int cpu_result = sum_array(in, SIZE);

    if (cpu_result == out) {
        printf("Good: %d\n", out);
    } else {
        printf("Bad, cpu: %d, gpu: %d \n", cpu_result, out);
    }
    free(in);
    return (cpu_result == out);
}