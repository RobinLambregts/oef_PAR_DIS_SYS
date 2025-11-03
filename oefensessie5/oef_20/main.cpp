#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <omp.h>
#include "timer.h"

using namespace std;

vector<float> readFloats(const string &fname)
{
    vector<float> data;
    ifstream f(fname, ios::binary);

    f.seekg(0, ios_base::end);
    int pos = f.tellg();
    f.seekg(0, ios_base::beg);
    if (pos <= 0)
        throw runtime_error("Can't seek in file " + fname + " or file has zero length");

    if (pos % sizeof(float) != 0)
        throw runtime_error("File " + fname + " doesn't contain an integer number of float32 values");

    int num = pos/sizeof(float);
    data.resize(num);

    f.read(reinterpret_cast<char*>(data.data()), pos);
    if (f.gcount() != pos)
        throw runtime_error("Incomplete read: " + to_string(f.gcount()) + " vs " + to_string(pos));
    return data;
}

pair<float, float> calculateExtremes(const vector<float> &data)
{
    float smallest = data[0];
    float biggest = data[0];
    for (float num : data)
    {
        if (num > biggest)
            biggest = num;
        if (num < smallest)
            smallest = num;
    }
    return make_pair(smallest, biggest);
}

vector<vector<float>> histogramOMP(const vector<float> &data, pair<float, float> extremes, int bins)
{
    float binWidth = (extremes.second - extremes.first) / bins;
    vector<int> counts(bins, 0);

    int nThreads = omp_get_max_threads();
    vector<vector<int>> localCounts(nThreads, vector<int>(bins, 0));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for
        for (size_t i = 0; i < data.size(); i++)
        {
            float val = data[i];
            if (val < extremes.first || val > extremes.second)
                continue;
            int idx = min(int((val - extremes.first) / binWidth), bins - 1);
            localCounts[tid][idx]++;
        }
    }

    for (int t = 0; t < nThreads; t++)
    {
        for (int b = 0; b < bins; b++)
        {
            counts[b] += localCounts[t][b];
        }
    }

    vector<vector<float>> hist(bins);
    for (int i = 0; i < bins; i++)
    {
        float start = extremes.first + i * binWidth;
        float end = start + binWidth;
        hist[i] = {start, end, static_cast<float>(counts[i])};
    }
    return hist;
}

void printhist(const vector<vector<float>> &histogram)
{
    for (int i = 0; i < histogram.size(); i++)
    {
        cout << "Bin " << i << " : ("
             << histogram[i][0] << " -> "
             << histogram[i][1] << ") "
             << "Count = " << histogram[i][2] << endl;
    }
}

int main()
{
    vector<float> data = readFloats("histvalues.dat");
    pair<float, float> extremes = calculateExtremes(data);

    cout << "Aantal bins N: ";
    int N;
    cin >> N;

    Timer timer;
    timer.start();
    vector<vector<float>> hist = histogramOMP(data, extremes, N);
    timer.stop();

    printhist(hist);
    double time_sec = timer.durationNanoSeconds() * 1e-9;
    cout << "Execution time (OpenMP histogram) = " << time_sec << " sec" << endl;

    return 0;
}
