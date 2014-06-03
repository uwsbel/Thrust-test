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

const int ARRAY_SIZE = 1000;

enum Order {
	ASCENDING,
	DESCENDING
};

enum Method {
	RAW,
	WRAPPED
};

// ------------------------------------------------------------------------------------

bool check_sort(Order order, double* hA)
{
	switch (order) {
	case ASCENDING:
		for (int i = 1; i < ARRAY_SIZE; i++) {
			if (hA[i] < hA[i-1])
				return false;
		}
		break;
	case DESCENDING:
		for (int i = 1; i < ARRAY_SIZE; i++) {
			if (hA[i] > hA[i-1])
				return false;
		}
		break;
	}

	return true;
}

bool sort_test(Order order, Method method)
{
	double* hA;
	hA = (double *) malloc (sizeof(double) * ARRAY_SIZE);
	for (int i = 0; i < ARRAY_SIZE; i++)
		hA[i] = 1.0 * (rand() % ARRAY_SIZE);

	double* dA;
	cudaMalloc((void **) &dA, sizeof(double) * ARRAY_SIZE);
	cudaMemcpy(dA, hA, sizeof(double) * ARRAY_SIZE, cudaMemcpyHostToDevice);

	switch (method) {
	case RAW:
		{
			switch (order) {
			case ASCENDING:
				thrust::sort(thrust::cuda::par, dA, dA + ARRAY_SIZE);
				break;
			case DESCENDING:
				//// NOTE: this seg-faults!!!
				thrust::sort(thrust::cuda::par, dA, dA + ARRAY_SIZE, thrust::greater<double>());
				break;
			}
			break;
		}
	case WRAPPED:
		{
			thrust::device_ptr<double> wdA = thrust::device_pointer_cast(dA);
			switch (order) {
			case ASCENDING:
				thrust::sort(wdA, wdA + ARRAY_SIZE);
				break;
			case DESCENDING:
				thrust::sort(wdA, wdA + ARRAY_SIZE, thrust::greater<double>());
				break;
			}
			break;
		}
	}

	cudaMemcpy(hA, dA, sizeof(double) * ARRAY_SIZE, cudaMemcpyDeviceToHost);
	bool result = check_sort(order, hA);

	cudaFree(dA);
	free(hA);

	return result;
}

// ------------------------------------------------------------------------------------

void sort_by_key_test(Order order, Method method)
{
	const int SIZE = 10;

	int hK[SIZE] = {0, 2, 1, 4, 2, 4, 0, 1, 4, 2};
	double hV[SIZE] = {8, 2, 8, 7, 8, 3, 5, 3, 7, 4};
	std::cout << "     ";
	for (int i = 0; i < SIZE; i++)
		std::cout << "(" << hK[i] << ", " << hV[i] << ") ";
	std::cout << std::endl;

	int* dK;
	double* dV;
	cudaMalloc((void **) &dK, sizeof(int) * SIZE);
	cudaMalloc((void **) &dV, sizeof(double) * SIZE);
	cudaMemcpy(dK, hK, sizeof(int) * SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dV, hV, sizeof(double) * SIZE, cudaMemcpyHostToDevice);

	switch (method) {
	case RAW:
		{
			switch (order) {
			case ASCENDING:
				thrust::sort_by_key(thrust::cuda::par, dK, dK + SIZE, dV);
				break;
			case DESCENDING:
				thrust::sort_by_key(thrust::cuda::par, dK, dK + SIZE, dV, thrust::greater<int>());
				break;
			}
			break;
		}
	case WRAPPED:
		{
			thrust::device_ptr<int> wdK = thrust::device_pointer_cast(dK);
			thrust::device_ptr<double> wdV = thrust::device_pointer_cast(dV);
			switch (order) {
			case ASCENDING:
				thrust::sort_by_key(wdK, wdK + SIZE, wdV);
				break;
			case DESCENDING:
				thrust::sort_by_key(wdK, wdK + SIZE, wdV, thrust::greater<int>());
				break;
			}
			break;
		}
	}

	cudaMemcpy(hK, dK, sizeof(int) * SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(hV, dV, sizeof(double) * SIZE, cudaMemcpyDeviceToHost);
	std::cout << "     ";
	for (int i = 0; i < SIZE; i++)
		std::cout << "(" << hK[i] << ", " << hV[i] << ") ";
	std::cout << std::endl;

	cudaFree(dK);
	cudaFree(dV);
}

// ------------------------------------------------------------------------------------

int main(int argc, char **argv)
{
	std::cout << "Sort ascending DR ...  " << std::flush << sort_test(ASCENDING, RAW) << std::endl;
	//std::cout << "Sort descending DR ... " << std::flush << sort_test(DESCENDING, RAW) << std::endl;

	std::cout << "Sort ascending DW ...  " << std::flush << sort_test(ASCENDING, WRAPPED) << std::endl;
	std::cout << "Sort descending DW ... " << std::flush << sort_test(DESCENDING, WRAPPED) << std::endl;

	std::cout << std::endl << std::endl;

	std::cout << "Sort_by_key ascending DR:" << std::endl;
	sort_by_key_test(ASCENDING, RAW);
	std::cout << "Sort_by_key descending DR:" << std::endl;
	sort_by_key_test(DESCENDING, RAW);

	std::cout << std::endl << std::endl;

	std::cout << "Sort_by_key ascending DW:" << std::endl;
	sort_by_key_test(ASCENDING, WRAPPED);
	std::cout << "Sort_by_key descending DW:" << std::endl;
	sort_by_key_test(DESCENDING, WRAPPED);

	return 0;
}
