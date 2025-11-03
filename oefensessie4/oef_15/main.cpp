#include <thread>
#include <random>
#include <iostream>
#include <chrono>

#include "som.h"

#define COUNT 100000
#define RANDOM_SLEEP_MS 10


#include "SafeQueue.h"
SafeQueue readerQueue;
SafeQueue sumQueue;






void readerThread(){
	std::cout << "Render Thread started." << std::endl;
	for (unsigned int i = 0; i < COUNT; ++i){
		std::this_thread::sleep_for(std::chrono::microseconds(rand() % RANDOM_SLEEP_MS));

		readerQueue.push(getallen[i]);

	}

}

void sumThread(){
	std::cout << "Sum Thread started." << std::endl;
	float sum = 0.0f;
	for (unsigned int i = 0; i < COUNT; ++i){
		std::this_thread::sleep_for(std::chrono::microseconds(rand() % RANDOM_SLEEP_MS));
		sum += readerQueue.get(0);
		readerQueue.pop();
		sumQueue.push(sum);
	}
}

void resultThread(){
	std::cout << "Results Thread started." << std::endl;
	float result = 0.0f;
	for (unsigned int i = 0; i < COUNT; ++i){
		std::this_thread::sleep_for(std::chrono::microseconds(rand() % RANDOM_SLEEP_MS)); // sleep for random amount of time in interval [0,1] seconds
		result = sumQueue.get(0);
		sumQueue.pop();

		if ((i + 1) % 5 == 0) // Print every 5 results
			std::cout << "Sum: " << result << std::endl;
	}
}


int main (int argc, char *argv[]) {
	std::cout << "Start SafeQueue." << std::endl;
	
	std::thread reader = std::thread(readerThread);
	std::thread sum = std::thread(sumThread);
	std::thread result = std::thread(resultThread);

	reader.join();
	std::cout << "Reader Thread Ended." << std::endl;
	sum.join();
	std::cout << "Sum Thread Ended." << std::endl;
	result.join();
	std::cout << "Result Thread Ended." << std::endl;
	
	
	float referenceSum = 0.0f;
	for (unsigned int i = 0; i < COUNT; ++i){
		referenceSum += getallen[i];
	}

	std::cout << "Reference Sum: " << referenceSum << std::endl;

}
