#include "image2d.h"
#include <stdlib.h>
#include <iostream>
#include <string>

using namespace std;

void Exit(const string &s)
{
	cerr << s << endl;
	exit(-1);
}

// This is the function you'll need to fill in, in fractal.cu
bool cudaFractal(int iterations, float xmin, float xmax, float ymin, float ymax, 
                 Image2D &output, string &errStr);

int main(int argc, char *argv[])
{
	if (argc != 7)
		Exit("Usage: " + string(argv[0]) + " outputimg_no_extension iterations xmin xmax ymin ymax\nSaves image to outputimg.png\n");

	string outputFile(argv[1]);
	int iterations = atoi(argv[2]);
	float xmin = atof(argv[3]);
	float xmax = atof(argv[4]);
	float ymin = atof(argv[5]);
	float ymax = atof(argv[6]);
	
	Image2D output;
	string errStr;

	if (!cudaFractal(iterations, xmin, xmax, ymin, ymax, output, errStr))
		Exit("CUDA Fractal program error: " + errStr);

	if (!output.exportPicture(outputFile, false, false, false)) // don't flip image and don't autoscale
		Exit("Can't export output: " + output.getErrorString());
		
	cout << "Program finished successfully" << endl;

	return 0;
}
