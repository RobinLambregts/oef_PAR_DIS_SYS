#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <utility>
#include <algorithm>
#include "timer.h"

using namespace std;

vector<float> readFloats(const string &fname)
{
    vector<float> data;
    ifstream f(fname, std::ios::binary);

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

vector<vector<float>> histogram(const vector<float> &data, pair<float, float> extremes, int bins)
{
    float binWidth = (extremes.second - extremes.first) / bins;
    vector counts(bins, 0);

    for (float val : data)
    {
        if (val < extremes.first || val > extremes.second)
            continue;
        int idx = min(int((val - extremes.first) / binWidth), bins - 1);
        counts[idx]++;
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

void timer(vector<int> &data, int N)
{
    float sum = 0;

    {
        AutoAverageTimer t("Row-major");
        for (int repeat = 0; repeat < 50; repeat++)
        {
            sum = 0;
            t.start();
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    sum += data[i * N + j];
                }
            }
            t.stop();
        }
    }

    {
        AutoAverageTimer t("Column-major");
        for (int repeat = 0; repeat < 50; repeat++)
        {
            sum = 0;
            t.start();
            for (int j = 0; j < N; j++)
            {
                for (int i = 0; i < N; i++)
                {
                    sum += data[i * N + j];
                }
            }
            t.stop();
        }
    }

    cout << "Sum = " << sum << endl;
}

int main()
{
    vector<float> data = readFloats("histvalues.dat");
    pair<float, float> extremes = calculateExtremes(data);

    cout << "waarde voor N (aantal bins): ";
    int N;
    cin >> N;

    vector<vector<float>> hist = histogram(data, extremes, N);
    printhist(hist);

    const int n = 20000;

    vector<int> intData(n*n);
    timer(intData, n);

    return 0;
}

//aanpassen naar int => gebeurt
//oO en o3 vergelijken + tekst schrijven over deze vergelijking
// => bevindt zich in output.txt
