#include <iostream>
#include <cstdint>
#include "timer.h"

using namespace std;

struct Entry {
    Entry* next;
    uint64_t padding[127];
};

void traverse(Entry* start) {
    const int steps = 20000000;
    Entry* current = start;

    Timer timer;
    timer.start();
    for (int i = 0; i < steps; ++i) {
        current = current->next;
    }
    timer.stop();

    double nsPerStep = timer.durationNanoSeconds() / steps;
    cout << nsPerStep << endl;
}

int main() {
    cout << "array size (int): ";
    int size;
    cin >> size;

    vector<Entry> entries(size);
    for (size_t i = 0; i < entries.size(); ++i) {
        entries[i].next = &entries[(i + 1) % entries.size()];
    }

    traverse(&entries[0]);
}
