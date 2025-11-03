#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

int main(int argc, char const *argv[])
 {
     std::vector<int> numbers = {243,3,4,51,234,455,76,2,4326,78,643};
     int min_val = numbers[0];

#pragma omp parallel for reduction(min:min_val)
    for (int number : numbers) {
        if (number < min_val) {
            min_val = number;
        }
    }

     printf("Minimum = %u\n", min_val);
     return 0;
 }
