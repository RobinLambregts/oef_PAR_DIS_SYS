#include <stdio.h>
#include <omp.h>

int main() {
#pragma omp parallel
    {
        int id = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        printf("Hello from thread %d van %d\n", id, nthreads);
    }

    return 0;
}
