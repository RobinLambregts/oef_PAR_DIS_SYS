#pragma once
#include <mutex>
#include <condition_variable>
#include <stdexcept>

class SafeQueue {
public:
	SafeQueue() {
		firstElement = 0;
		lastElement = 0;
	}

	~SafeQueue() {
		firstElement = 0;
		lastElement = 0;
	}

	void push(float f) {
		std::unique_lock lock(mtx);
		std::cout << "push: " << f << std::endl;
		queue_not_full.wait(lock, [this] { return lastElement < queueSize; });
		queue[lastElement] = f;
		++lastElement;
		queue_not_empty.notify_one();
	}

	void pop() {
		std::unique_lock lock(mtx);
		std::cout << "pop: " << queue[firstElement] << std::endl;
		queue_not_empty.wait(lock, [this] { return lastElement > 0; });
		--lastElement;
		queue_not_full.notify_one();
	}

	float get(unsigned int i) const {
		std::unique_lock lock(mtx);
		float ret = queue[(firstElement+i) % queueSize];
		return ret;
	}

	unsigned int getSize() {
		std::unique_lock lock(mtx);
		unsigned int tmp = lastElement;
		if(lastElement < firstElement)
			tmp += queueSize;
		tmp = tmp-firstElement;
		return tmp;
	}

private:
	unsigned int firstElement, lastElement;

	const static unsigned int queueSize = 2;
	float queue[queueSize];
	mutable std::mutex mtx;
	std::condition_variable queue_not_empty;
	std::condition_variable queue_not_full;
};
