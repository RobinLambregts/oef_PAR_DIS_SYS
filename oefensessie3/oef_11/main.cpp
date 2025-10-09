#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <utility>
#include <algorithm>

using namespace std;

vector<int> localHistogram(const vector<float> &localData, float minVal, float maxVal, int bins) {
    vector<int> counts(bins, 0);
    float binWidth = (maxVal - minVal) / bins;

    for (float val : localData) {
        if (val < minVal || val > maxVal)
            continue;
        int idx = min(int((val - minVal) / binWidth), bins - 1);
        counts[idx]++;
    }
    return counts;
}

int main() {
    MPI_Init(nullptr, nullptr);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<float> data;
    int totalSize = 0;

    if (rank == 0) {
        ifstream f("histvalues.dat", ios::binary);
        if (!f) {
            cerr << "Kan bestand niet openen!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        f.seekg(0, ios_base::end);
        int pos = f.tellg();
        f.seekg(0, ios_base::beg);

        if (pos % sizeof(float) != 0) {
            cerr << "Bestand bevat geen integer aantal floats!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        totalSize = pos / sizeof(float);
        if (totalSize % size != 0) {
            cerr << "Aantal getallen (" << totalSize << ") is niet deelbaar door aantal processen (" << size << ")" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        data.resize(totalSize);
        f.read(reinterpret_cast<char*>(data.data()), pos);
        if (f.gcount() != pos) {
            cerr << "Onvolledige leesactie!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&totalSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int chunkSize = totalSize / size;
    vector<float> localData(chunkSize);

    MPI_Scatter(data.data(), chunkSize, MPI_FLOAT,
                localData.data(), chunkSize, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    float localMin = *min_element(localData.begin(), localData.end());
    float localMax = *max_element(localData.begin(), localData.end());
    float globalMin, globalMax;

    MPI_Allreduce(&localMin, &globalMin, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&localMax, &globalMax, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

    int bins = 10;
    if (rank == 0) {
        cout << "Aantal bins: ";
        cin >> bins;
    }
    MPI_Bcast(&bins, 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> localCounts = localHistogram(localData, globalMin, globalMax, bins);

    vector<int> globalCounts(bins, 0);
    MPI_Reduce(localCounts.data(), globalCounts.data(), bins, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        float binWidth = (globalMax - globalMin) / bins;
        for (int i = 0; i < bins; ++i) {
            float start = globalMin + i * binWidth;
            float end = start + binWidth;
            cout << "Bin " << i << " (" << start << " -> " << end << ") = " << globalCounts[i] << endl;
        }
    }

    MPI_Finalize();
    return 0;
}
