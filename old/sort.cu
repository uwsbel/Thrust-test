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

void sort_test(bool decrease = false) {
	cout << "Sort ";
	if (decrease)
		cout << "non-increasingly ... " << flush;
	else
		cout << "non-decreasingly ... " << flush;

	const int ARRAY_SIZE = 1000;

	double *hA, *dA;

	hA = (double *) malloc (sizeof(double) * ARRAY_SIZE);
	cudaMalloc(&dA, sizeof(double) * ARRAY_SIZE);

	for (int i = 0; i < ARRAY_SIZE; i++)
		hA[i] = 1.0 * (rand() % ARRAY_SIZE);

	cudaMemcpy(dA, hA, sizeof(double) * ARRAY_SIZE, cudaMemcpyHostToDevice);

	if (decrease) {
		thrust::device_ptr<double> dA_begin = thrust::device_pointer_cast(dA);
		thrust::device_ptr<double> dA_end   = thrust::device_pointer_cast(dA + ARRAY_SIZE);
		thrust::sort(thrust::cuda::par, dA_begin, dA_end, thrust::greater<double>());

		/* FIXME: The program gets seg-fault if we do:
		   thrust::sort(thrust::cuda::par, dA, dA + ARRAY_SIZE, thrust::greater<double>());
		   which does not seem to make much sense.
		   */
	}
	else
		thrust::sort(thrust::cuda::par, dA, dA + ARRAY_SIZE);

	cudaMemcpy(hA, dA, sizeof(double) * ARRAY_SIZE, cudaMemcpyDeviceToHost);

	bool correct = true;
	for (int i = 1; i < ARRAY_SIZE; i++)
		if (decrease ? (hA[i] > hA[i-1]) : (hA[i] < hA[i-1])) {
			correct = false;
			break;
		}

	cudaFree(dA);
	free(hA);

	if (correct)
		cout << "OK" << endl;
	else
		cout << "Failed" << endl;
}

void sort_by_key_test(bool decrease = false) {
	cout << "Sort by key ";
	if (decrease)
		cout << "non-increasingly ... " << endl;
	else
		cout << "non-decreasingly ... " << endl;

	const int ARRAY_SIZE = 10;

	int    *h_keys,   *d_keys;
	double *h_values, *d_values;

	h_keys = (int *)malloc(sizeof(int) * ARRAY_SIZE);
	h_values = (double *)malloc(sizeof(double) * ARRAY_SIZE);

	cudaMalloc(&d_keys, sizeof(int) * ARRAY_SIZE);
	cudaMalloc(&d_values, sizeof(double) * ARRAY_SIZE);

	cout << "Before: " << endl;
	for (int i = 0; i < ARRAY_SIZE; i++) {
		h_keys[i] = rand() % (ARRAY_SIZE >> 1);
		h_values[i] = 1.0 * (rand() % ARRAY_SIZE);
		cout << "(" << h_keys[i] << ", " << h_values[i] << ") ";
	}
	cout << endl;

	cudaMemcpy(d_keys, h_keys, sizeof(int) * ARRAY_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_values, h_values, sizeof(double) * ARRAY_SIZE, cudaMemcpyHostToDevice);

	if (decrease) {
		thrust::device_ptr<int> keys_begin = thrust::device_pointer_cast(d_keys);
		thrust::device_ptr<int> keys_end   = thrust::device_pointer_cast(d_keys + ARRAY_SIZE);
		thrust::device_ptr<double> values_begin = thrust::device_pointer_cast(d_values);
		thrust::sort_by_key(thrust::cuda::par, keys_begin, keys_end, values_begin, thrust::greater<int>());
		/* FIXME: The program gets wrong results if we do:
		   thrust::sort_by_key(thrust::cuda::par, d_keys, d_keys + ARRAY_SIZE, d_values, thrust::greater<int>());
		   which does not seem to make much sense.
		   */
	}
	else
		thrust::sort_by_key(thrust::cuda::par, d_keys, d_keys + ARRAY_SIZE, d_values);

	cudaMemcpy(h_keys, d_keys, sizeof(int) * ARRAY_SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_values, d_values, sizeof(double) * ARRAY_SIZE, cudaMemcpyDeviceToHost);

	cout << "After: " << endl;
	for (int i = 0; i < ARRAY_SIZE; i++) {
		cout << "(" << h_keys[i] << ", " << h_values[i] << ") ";
	}
	cout << endl;

	cudaFree(d_keys);
	cudaFree(d_values);
	free(h_keys);
	free(h_values);
}

int main(int argc, char **argv)
{
	sort_test(false);
	sort_test(true);
	sort_by_key_test(false);
	sort_by_key_test(true);
	return 0;
}
