#pragma once
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <iostream>

class SafeQueue {
public:
    SafeQueue() : firstElement(0), lastElement(0), currentSize(0) {}

    ~SafeQueue() = default;

    // Voeg een element toe, wacht als de queue vol is
    void push(float f) {
        std::unique_lock lock(mtx);
        queue_not_full.wait(lock, [this] { return currentSize < queueSize; });

        queue[lastElement] = f;
        lastElement = (lastElement + 1) % queueSize;
        ++currentSize;

        queue_not_empty.notify_all(); // notify alle lezers
    }

    // Verwijder het eerste element uit de queue, wacht als leeg
    float pop() {
        std::unique_lock lock(mtx);
        queue_not_empty.wait(lock, [this] { return currentSize > 0; });

        float val = queue[firstElement];
        firstElement = (firstElement + 1) % queueSize;
        --currentSize;

        queue_not_full.notify_one();
        return val;
    }

    // Ophalen van het i-de element, wacht totdat beschikbaar
    float waitAndGet(unsigned int i = 0) {
        std::unique_lock lock(mtx);
        queue_not_empty.wait(lock, [this, i] { return i < currentSize; });
        return queue[(firstElement + i) % queueSize];
    }

    // Read-only functie: i-de element ophalen, wacht als nog niet beschikbaar
    float get(unsigned int i) const {
        std::shared_lock lock(mtx); // meerdere lezers tegelijk toegestaan
        queue_not_empty.wait(lock, [this, i] { return i < currentSize; });
        return queue[(firstElement + i) % queueSize];
    }

    // Read-only functie: huidige grootte van de queue
    unsigned int getSize() const {
        std::shared_lock lock(mtx);
        return currentSize;
    }

private:
    unsigned int firstElement, lastElement;
    unsigned int currentSize;

    const static unsigned int queueSize = 1024; // aanpasbare grootte
    float queue[queueSize];

    mutable std::shared_mutex mtx;
    mutable std::condition_variable_any queue_not_empty;
    mutable std::condition_variable_any queue_not_full;
};
