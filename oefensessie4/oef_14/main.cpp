#include <iostream>
#include <vector>
#include <thread>
#include <cmath>
#include <mutex>
#include "timer.h"

double calculateArea(double bin, double binsize) {
    return sqrt(1 - bin*bin) * binsize;
}

int main() {
    double min = -1.0, max = 1.0;
    long N = 1000000;
    double binsize = (max - min) / N;

    unsigned int numThreads = std::thread::hardware_concurrency();
    if(numThreads == 0) numThreads = 4; // fallback

    std::vector<std::thread> threads(numThreads);
    double totalSum = 0.0;
    std::mutex sumMutex; // beschermt totale som bij gelijktijdige writes

    long chunk = N / numThreads;

    AutoAverageTimer tAcc("Threaded Accumulate");
    tAcc.start();

    for(unsigned int t = 0; t < numThreads; ++t) {
        long start = t * chunk;
        long end = (t == numThreads - 1) ? N : start + chunk;

        threads[t] = std::thread([&, start, end]() {
            double localSum = 0.0;
            for(long i = start; i < end; ++i) {
                double bin = min + i * binsize;
                localSum += calculateArea(bin, binsize);
            }
            // directe accumulatie naar gedeelde variabele
            std::lock_guard<std::mutex> guard(sumMutex);
            totalSum += localSum;
        });
    }

    for(auto& th : threads) th.join();

    tAcc.stop();

    std::cout << "Threaded Accumulate calculated area: " << totalSum << std::endl;
    std::cout << "Actual area: " << M_PI / 2 << std::endl;
    tAcc.report(std::cout);

    return 0;
}

/*
* OUTPUT:
*
"C:\Users\robin\Documents\4e jaar\PAR_DIS_SYS\main.exe"
Threaded Accumulate calculated area: 1.5708
Actual area: 1.5708
#Threaded Accumulate 0.0130276 +/- 0 sec (1 measurements)

Process finished with exit code 0

VERGELIJKING:
Beide thread-gebaseerde versies berekenen correct de oppervlakte van 1.5708 (
𝜋
/
2
π/2). Threaded Calculation laat elke thread zijn lokale som opslaan en telt ze achteraf bij elkaar op,
zonder mutex, wat resulteert in de snelste uitvoering (0.0082 s).
Threaded Accumulate voegt de lokale sommen direct toe aan een gedeelde variabele met mutex,
wat correct maar iets trager is (0.0130 s) door synchronisatie-overhead.
Functioneel identiek, maar lock-free Threaded Calculation is efficiënter.
 * */