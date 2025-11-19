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
    std::vector<double> localSums(numThreads, 0.0);

    long chunk = N / numThreads;

    AutoAverageTimer tCalc("Threaded Calculation");
    tCalc.start();

    for(unsigned int t = 0; t < numThreads; ++t) {
        long start = t * chunk;
        long end = (t == numThreads - 1) ? N : start + chunk;

        threads[t] = std::thread([&, start, end, t]() {
            double localSum = 0.0;
            for(long i = start; i < end; ++i) {
                double bin = min + i * binsize;
                localSum += calculateArea(bin, binsize);
            }
            localSums[t] = localSum; // schrijf naar vector op index t
        });
    }

    for(auto& th : threads) th.join();

    double totalSum = 0.0;
    for(double s : localSums) totalSum += s;

    tCalc.stop();

    std::cout << "Threaded calculated area: " << totalSum << std::endl;
    std::cout << "Actual area: " << M_PI / 2 << std::endl;
    tCalc.report(std::cout);

    return 0;
}

/*
* OUTPUT:
*
"C:\Users\robin\Documents\4e jaar\PAR_DIS_SYS\main.exe"
Threaded calculated area: 1.5708
Actual area: 1.5708
#Threaded Calculation 0.0082125 +/- 0 sec (1 measurements)

Process finished with exit code 0
 * */