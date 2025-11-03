#include <cmath>
#include <cstdio>
#include <omp.h>
#include "timer.h"

bool is_prime(size_t number)
{
    if (number < 2) return false;
    size_t to_check = std::sqrt(number) + 1;
    for (size_t i = 2; i < to_check; ++i)
    {
        if (number % i == 0)
            return false;
    }
    return true;
}

int main()
{
    const size_t start = 2;
    const size_t end = 100000000;

    int prime_count = 0;
    Timer t;
    t.start();

#pragma omp parallel for reduction(+:prime_count) schedule(guided)
    for (size_t i = start; i <= end; ++i) {
        if (is_prime(i))
            prime_count += 1;
    }

    t.stop();
    double time_sec = t.durationNanoSeconds() * 1e-9;
    printf("Schedule=guided | Primes=%d | Tijd=%.6f sec\n", prime_count, time_sec);


    return 0;
}

/*
Schedule=static | Primes=5761455 | Tijd=114.256742 sec
Schedule=dynamic | Primes=5761455 | Tijd=114.297017 sec
Schedule=guided | Primes=5761455 | Tijd=152.777893 sec

De resultaten laten zien dat alle drie de schedules correct hetzelfde
aantal priemgetallen opleveren. Qua prestaties presteert static iets beter dan dynamic,
terwijl guided duidelijk trager is voor deze workload.
Dit komt doordat de rekentijd per iteratie relatief uniform is,
waardoor statische verdeling het minste overhead heeft en
guided extra chunk-berekeningen introduceert zonder voordeel.
 */
