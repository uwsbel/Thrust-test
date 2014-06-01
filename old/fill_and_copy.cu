#include <iostream>
#include <cmath>

#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/gather.h>
#include <thrust/logical.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>

using std::cout;
using std::cerr;
using std::endl;
using std::flush;

void fill_test() {
	cout << "Fill test ... " << flush;
	const int ARRAY_SIZE = 1000;

	double *hA, *dA;

	hA = (double *)malloc(sizeof(double) * ARRAY_SIZE);
	cudaMalloc(&dA, sizeof(double) * ARRAY_SIZE);

	for (int i = 0; i < ARRAY_SIZE; i++)
		hA[i] = 0.0;

	cudaMemcpy(dA, hA, sizeof(double) * ARRAY_SIZE, cudaMemcpyHostToDevice);
	thrust::fill(thrust::cuda::par, dA, dA + ARRAY_SIZE, 9.0);
	cudaMemcpy(hA, dA, sizeof(double) * ARRAY_SIZE, cudaMemcpyDeviceToHost);

	bool correct = true;
	for (int i = 0; i < ARRAY_SIZE; i++)
		if (hA[i] != 9.0) {
			correct = false;
			break;
		}

	free(hA);
	cudaFree(dA);

	if (correct)
		cout << "OK" << endl;
	else
		cout << "Failed" << endl;

}

void copy_test() {
	cout << "Copy test ... " << flush;

	const int ARRAY_SIZE = 1000;

	double *hA, *dA, *dB, *hB;

	hA = (double *)malloc(sizeof(double) * ARRAY_SIZE);
	hB = (double *)malloc(sizeof(double) * ARRAY_SIZE);
	cudaMalloc(&dA, sizeof(double) * ARRAY_SIZE);
	cudaMalloc(&dB, sizeof(double) * ARRAY_SIZE);

	for (int i = 0; i < ARRAY_SIZE; i++) {
		hA[i] = 1.0 * (i+1);
		hB[i] = 0.0;
	}

	cudaMemcpy(dA, hA, sizeof(double) * ARRAY_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, sizeof(double) * ARRAY_SIZE, cudaMemcpyHostToDevice);
	thrust::copy(thrust::cuda::par, dA, dA + ARRAY_SIZE, dB);
	cudaMemcpy(hB, dB, sizeof(double) * ARRAY_SIZE, cudaMemcpyDeviceToHost);

	bool correct = true;
	for (int i = 0; i < ARRAY_SIZE; i++)
		if (hB[i] != 1.0 * (i + 1)) {
			correct = false;
			break;
		}

	if (correct)
		cout << "OK" << endl;
	else
		cout << "Failed" << endl;

	free(hA);
	cudaFree(dA);
	free(hB);
	cudaFree(dB);
}

void sequence_test() {
	cout << "Sequence test ... " << flush;

	const int ARRAY_SIZE = 1000;

	double *hA, *dA;

	hA = (double *)malloc(sizeof(double) * ARRAY_SIZE);
	cudaMalloc(&dA, sizeof(double) * ARRAY_SIZE);

	for (int i = 0; i < ARRAY_SIZE; i++)
		hA[i] = 0.0;

	/* FIXME: it's not correct to call
	    thrust::sequence(thrust::cuda::par, dA, dA + ARRAY_SIZE);
	   */
	{
		thrust::device_ptr<double> A_begin(dA);
		thrust::device_ptr<double> A_end(dA + ARRAY_SIZE);
		thrust::sequence(thrust::cuda::par, A_begin, A_end);
	}
	cudaMemcpy(hA, dA, sizeof(double) * ARRAY_SIZE, cudaMemcpyDeviceToHost);

	bool correct = true;
	for (int i = 0; i < ARRAY_SIZE; i++)
		if (hA[i] != 1.0 * i) {
			correct = false;
			break;
		}

	if (correct)
		cout << "OK" << endl;
	else
		cout << "Failed" << endl;

	free(hA);
	cudaFree(dA);
}

int main(int argc, char **argv) 
{
	fill_test();
	copy_test();
	sequence_test();
	return 0;
}
