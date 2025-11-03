#include <cstdio>
#include <cstdint>
#include <vector>
#include <omp.h>
#include "timer.h"

const int64_t g_numLoops = 1<<27;

void f(uint8_t *pBuffer, int offset)
{
    for (int64_t i = 0 ; i < g_numLoops ; i++)
        pBuffer[offset] += 1;
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        printf("Gebruik: %s <multiplier>\n", argv[0]);
        return 1;
    }

    int multiplier = atoi(argv[1]);
    const int nThreads = 4;

    std::vector<uint8_t> buffer(multiplier * nThreads + 1, 0);

    Timer timer;
    timer.start();

#pragma omp parallel num_threads(nThreads)
    {
        int id = omp_get_thread_num();
        int offset = multiplier * id;
        f(buffer.data(), offset);
    }

    timer.stop();

    double time_sec = timer.durationNanoSeconds() * 1e-9;
    printf("Multiplier=%d | Tijd=%.6f sec\n", multiplier, time_sec);

    return 0;
}

/*
 * Multiplier=1 | Tijd=0.286312 sec
 * Multiplier=2 | Tijd=0.247638 sec
 * Multiplier=4 | Tijd=0.262636 sec
 * Multiplier=8 | Tijd=0.254938 sec
 * Multiplier=16 | Tijd=0.260177 sec
 * Multiplier=32 | Tijd=0.265591 sec
 * Multiplier=128 | Tijd=0.243607 sec
 * Multiplier=256 | Tijd=0.265674 sec
 *
 * Uit de metingen blijkt duidelijk dat de afstand tussen de geheugenlocaties
 * die door verschillende threads worden bewerkt een grote invloed heeft op de prestaties.
 * Kleine multipliers veroorzaken false sharing,
 * waardoor threads elkaar storen en de uitvoering vertraagt.
 * Bij grotere multipliers werken de threads op aparte cache lines,
 * verdwijnen de conflicten en stabiliseert de uitvoeringstijd.
 * Dit benadrukt het belang van cachevriendelijk geheugenbeheer bij parallelle programma’s.
 */
