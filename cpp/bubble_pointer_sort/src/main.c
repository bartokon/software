#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DEBUG 0
#define _DEBUG_PRINT \
    if (DEBUG) printf("%d -> %s\n", __LINE__, __FILE__);

static size_t sort_calls = 0u;

typedef struct number {
    unsigned int *val;
    struct number *next;
    struct number *prev;
    void (*printf_number)(struct number*);
} number;

void printf_number(number *number) {
    printf("Val is: %.2d -> %p prev: %p, this: %p, next: %p \n",
        *number->val,
        number->val,
        number->prev,
        number,
        number->next
    );
}

//Sort asecending
void bubble_sort_next(
    number *current,
    unsigned int *swapped
) {
    number *next = current->next;
    // Check if the current node is the last node
    if (next == NULL) return;
    if (*current->val > *next->val) {
        // Swap the pointers of the current and next nodes
        _DEBUG_PRINT
        unsigned int *temp = current->val;
        current->val = next->val;
        next->val = temp;
        *swapped += 1;
    }
    bubble_sort_next(next, swapped);
}

void print_list(number *nodes) {
    nodes->printf_number(nodes);
    if (nodes->next != NULL) print_list(nodes->next);
}

void sort_list(number *nodes) {
    unsigned int swapped = 0u;
    bubble_sort_next(nodes, &swapped);
    printf("Swapped %d times, calls %ld\n", swapped, ++sort_calls);
    if (swapped != 0u) sort_list(nodes);
    return;
}

int main (void) {
    printf("Hello World!\n");
    srand(time(NULL));

    size_t const NUMBER_OF_STRUCTS = 16;
    void *number_memory = malloc(NUMBER_OF_STRUCTS * sizeof(number));
    if (number_memory == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    unsigned int *number_values = (unsigned int*)malloc(NUMBER_OF_STRUCTS * sizeof(unsigned int));
    if (number_values == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    printf("Creating %ld structs\n", NUMBER_OF_STRUCTS);
    generate_numbers: for (unsigned int i = 0; i < NUMBER_OF_STRUCTS; ++i) {
        number *number_at_index = (number*)(number_memory);
        number_values[i] = rand() % 100 + 1;
        number_at_index[i].val = &number_values[i];
        number_at_index[i].next = i == NUMBER_OF_STRUCTS - 1 ? NULL : &number_at_index[i+1];
        number_at_index[i].prev = i == 0 ? NULL : &number_at_index[i-1];
        number_at_index[i].printf_number = &printf_number;
    }

    printf("Printing list before sorting\n");
    print_list((number*)number_memory);
    printf("Calling sort function\n");
    sort_list((number*)number_memory);
    printf("Printing list after sorting\n");
    print_list((number*)number_memory);

    free(number_memory);
    free(number_values);
    return 0;
}
