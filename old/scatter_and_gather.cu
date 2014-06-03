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

const int    ARRAY_SIZE = 10;
const double PRESET_VALUE = 10000.0;

enum Method {
	RAW,
	WRAPPED
};

bool check_scatter(const double *h_output, const double *h_input, const int *h_map)
{
	for (int i = 0; i < ARRAY_SIZE; i++)
		if (h_output[h_map[i]] != h_input[i])
			return false;

	return true;
}

bool scatter_test(Method method) {
	double *h_input, *d_input;
	double *h_output, *d_output;
	int    *h_map,  *d_map;

	h_input  = (double *)malloc (sizeof(double) * ARRAY_SIZE);
	h_output = (double *)malloc (sizeof(double) * ARRAY_SIZE);
	h_map    = (int *)   malloc (sizeof(int)    * ARRAY_SIZE);
	cudaMalloc(&d_input, sizeof(double) * ARRAY_SIZE);
	cudaMalloc(&d_output , sizeof(double) * ARRAY_SIZE);
	cudaMalloc(&d_map, sizeof(int) * ARRAY_SIZE);

	h_map[0] = 9;
	h_map[1] = 6;
	h_map[2] = 8;
	h_map[3] = 0;
	h_map[4] = 4;
	h_map[5] = 2;
	h_map[6] = 3;
	h_map[7] = 7;
	h_map[8] = 5;
	h_map[9] = 1;

	for (int i = 0; i < ARRAY_SIZE; i++) {
		h_input[i] = 10.0 + (i + 1);
		h_output[i] = PRESET_VALUE;
	}

	cudaMemcpy(d_input, h_input, sizeof(double) * ARRAY_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_output, h_output, sizeof(double) * ARRAY_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_map,   h_map,   sizeof(int) * ARRAY_SIZE,    cudaMemcpyHostToDevice);

	switch(method) {
	case RAW:
		thrust::scatter(thrust::cuda::par, d_input, d_input + ARRAY_SIZE, d_map, d_output);
		break;
	case WRAPPED:
		{
			thrust::device_ptr<double> wdInput(d_input), wdOutput(d_output);
			thrust::device_ptr<int>    wdMap(d_map);
			thrust::scatter(thrust::cuda::par, wdInput, wdInput + ARRAY_SIZE, wdMap, wdOutput);
			break;
		}
	default: break;
	}

	cudaMemcpy(h_output, d_output, sizeof(double) * ARRAY_SIZE, cudaMemcpyDeviceToHost);

	bool result = check_scatter(h_output, h_input, h_map);

	cudaFree(d_map);
	cudaFree(d_input);
	cudaFree(d_output);
	free(h_map);
	free(h_output);
	free(h_input);

	return result;
}

bool check_scatter_if(const double *h_output, const double *h_input, const int *h_map, const bool *h_stencil)
{
	bool   h_output_visited[ARRAY_SIZE] = {0};

	for (int i = 0; i < ARRAY_SIZE; i++)
		if (h_stencil[i]) {
			h_output_visited[h_map[i]] = true;
			if (h_output[h_map[i]] != h_input[i])
				return false;
		}

	for (int i = 0; i < ARRAY_SIZE; i++)
		if (!h_output_visited[i]) {
			if (h_output[i] != PRESET_VALUE)
				return false;
		}

	return true;
}

bool scatter_if_test(Method method) {
	double *h_input, *d_input;
	double *h_output, *d_output;
	int    *h_map,  *d_map;
	bool   *h_stencil, *d_stencil;

	h_input  = (double *)malloc (sizeof(double) * ARRAY_SIZE);
	h_output = (double *)malloc (sizeof(double) * ARRAY_SIZE);
	h_map    = (int *)   malloc (sizeof(int)    * ARRAY_SIZE);
	h_stencil = (bool *) malloc (sizeof(bool)   * ARRAY_SIZE);
	cudaMalloc(&d_input, sizeof(double) * ARRAY_SIZE);
	cudaMalloc(&d_output , sizeof(double) * ARRAY_SIZE);
	cudaMalloc(&d_map, sizeof(int) * ARRAY_SIZE);
	cudaMalloc(&d_stencil, sizeof(bool) * ARRAY_SIZE);

	h_map[0] = 9;
	h_map[1] = 6;
	h_map[2] = 8;
	h_map[3] = 0;
	h_map[4] = 4;
	h_map[5] = 2;
	h_map[6] = 3;
	h_map[7] = 7;
	h_map[8] = 5;
	h_map[9] = 1;

	for (int i = 0; i < ARRAY_SIZE; i++) {
		h_input[i] = 10.0 + (i + 1);
		h_stencil[i] = ((i & 1) ? true : false);
		h_output[i] = PRESET_VALUE;
	}

	cudaMemcpy(d_input, h_input, sizeof(double) * ARRAY_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_output, h_output, sizeof(double) * ARRAY_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_stencil, h_stencil, sizeof(bool) * ARRAY_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_map,   h_map,   sizeof(int) * ARRAY_SIZE,    cudaMemcpyHostToDevice);

	switch(method)
	{
	case RAW:
		thrust::scatter_if(thrust::cuda::par, d_input, d_input + ARRAY_SIZE, d_map, d_stencil, d_output);
		break;
	case WRAPPED:
		{
			thrust::device_ptr<double> wdInput(d_input), wdOutput(d_output);
			thrust::device_ptr<int>    wdMap(d_map);
			thrust::device_ptr<bool>   wdStencil(d_stencil);
			thrust::scatter_if(thrust::cuda::par, wdInput, wdInput + ARRAY_SIZE, wdMap, wdStencil, wdOutput, thrust::identity<bool>());
			break;
		}
	default: break;
	}

	cudaMemcpy(h_output, d_output, sizeof(double) * ARRAY_SIZE, cudaMemcpyDeviceToHost);

	bool result = check_scatter_if(h_output, h_input, h_map, h_stencil);

	cudaFree(d_map);
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_stencil);
	free(h_map);
	free(h_output);
	free(h_input);
	free(h_stencil);

	return result;
}

bool check_gather(const double *h_output, const double *h_input, const int *h_map)
{
	for (int i = 0; i < ARRAY_SIZE; i++)
		if (h_output[i] != h_input[h_map[i]])
			return false;

	return true;
}

bool gather_test(Method method) {
	double *h_input, *d_input;
	double *h_output, *d_output;
	int    *h_map,  *d_map;

	h_input  = (double *)malloc (sizeof(double) * ARRAY_SIZE);
	h_output = (double *)malloc (sizeof(double) * ARRAY_SIZE);
	h_map    = (int *)   malloc (sizeof(int)    * ARRAY_SIZE);
	cudaMalloc(&d_input, sizeof(double) * ARRAY_SIZE);
	cudaMalloc(&d_output , sizeof(double) * ARRAY_SIZE);
	cudaMalloc(&d_map, sizeof(int) * ARRAY_SIZE);

	h_map[0] = 9;
	h_map[1] = 6;
	h_map[2] = 8;
	h_map[3] = 0;
	h_map[4] = 4;
	h_map[5] = 2;
	h_map[6] = 3;
	h_map[7] = 7;
	h_map[8] = 5;
	h_map[9] = 1;

	for (int i = 0; i < ARRAY_SIZE; i++) {
		h_input[i]  = 10.0 + (i + 1);
		h_output[i] = PRESET_VALUE;
	}

	cudaMemcpy(d_input, h_input, sizeof(double) * ARRAY_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_output, h_output, sizeof(double) * ARRAY_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_map,   h_map,   sizeof(int) * ARRAY_SIZE,    cudaMemcpyHostToDevice);

	switch (method) {
	case RAW:
		thrust::gather(thrust::cuda::par, d_map, d_map + ARRAY_SIZE, d_input, d_output);
		break;
	case WRAPPED:
		{
			thrust::device_ptr<double> wdInput(d_input), wdOutput(d_output);
			thrust::device_ptr<int>    wdMap(d_map);
			thrust::gather(thrust::cuda::par, wdMap, wdMap + ARRAY_SIZE, wdInput, wdOutput);
			break;
		}
	default: break;
	}

	cudaMemcpy(h_output, d_output, sizeof(double) * ARRAY_SIZE, cudaMemcpyDeviceToHost);

	bool result = check_gather(h_output, h_input, h_map);

	cudaFree(d_map);
	cudaFree(d_input);
	cudaFree(d_output);
	free(h_map);
	free(h_output);
	free(h_input);

	return result;
}

bool check_gather_if(const double *hOutput, const double *hInput, const int * hMap, const bool *hStencil)
{
	for (int i = 0; i < ARRAY_SIZE; i++) {
		if (hStencil[i]) {
			if (hOutput[i] != hInput[hMap[i]]) 
				return false;
		} else {
			if (hOutput[i] != PRESET_VALUE)
				return false;
		}
	}
	return true;
}

bool gather_if_test(Method method) {
	double *h_input, *d_input;
	double *h_output, *d_output;
	int    *h_map,  *d_map;
	bool   *h_stencil, *d_stencil;

	h_input  = (double *)malloc (sizeof(double) * ARRAY_SIZE);
	h_output = (double *)malloc (sizeof(double) * ARRAY_SIZE);
	h_map    = (int *)   malloc (sizeof(int)    * ARRAY_SIZE);
	h_stencil = (bool *) malloc (sizeof(bool)   * ARRAY_SIZE);
	cudaMalloc(&d_input, sizeof(double) * ARRAY_SIZE);
	cudaMalloc(&d_output , sizeof(double) * ARRAY_SIZE);
	cudaMalloc(&d_map, sizeof(int) * ARRAY_SIZE);
	cudaMalloc(&d_stencil, sizeof(bool) * ARRAY_SIZE);

	h_map[0] = 9;
	h_map[1] = 6;
	h_map[2] = 8;
	h_map[3] = 0;
	h_map[4] = 4;
	h_map[5] = 2;
	h_map[6] = 3;
	h_map[7] = 7;
	h_map[8] = 5;
	h_map[9] = 1;

	for (int i = 0; i < ARRAY_SIZE; i++) {
		h_input[i] = 10.0 + (i + 1);
		h_stencil[i] = ((i & 1) ? true : false);
		h_output[i] = PRESET_VALUE;
	}

	cudaMemcpy(d_input,  h_input,   sizeof(double) * ARRAY_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_output, h_output,  sizeof(double) * ARRAY_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_stencil, h_stencil, sizeof(bool) * ARRAY_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_map,    h_map,     sizeof(int) * ARRAY_SIZE,    cudaMemcpyHostToDevice);

	switch (method)
	{
	case RAW:
		thrust::scatter_if(thrust::cuda::par, d_input, d_input + ARRAY_SIZE, d_map, d_stencil, d_output);
		break;
	case WRAPPED:
		{
			thrust::device_ptr<double> wdInput(d_input), wdOutput(d_output);
			thrust::device_ptr<int>    wdMap(d_map);
			thrust::device_ptr<bool>   wdStencil(d_stencil);
			thrust::gather_if(thrust::cuda::par, wdMap, wdMap + ARRAY_SIZE, wdStencil, wdInput, wdOutput, thrust::identity<bool>());
			break;
		}
	default: break;
	}

	cudaMemcpy(h_output, d_output, sizeof(double) * ARRAY_SIZE, cudaMemcpyDeviceToHost);

	bool result = check_gather_if(h_output, h_input, h_map, h_stencil);

	cudaFree(d_map);
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_stencil);
	free(h_map);
	free(h_output);
	free(h_input);
	free(h_stencil);

	return result;
}

int main(int argc, char **argv) 
{
	std::cout << "Scatter DR ... " << std::flush << scatter_test(RAW) << std::endl;
	std::cout << "Scatter DW ... " << std::flush << scatter_test(WRAPPED) << std::endl;
	std::cout << "Scatter_if DR ... " << std::flush << scatter_if_test(RAW) << std::endl;
	std::cout << "Scatter_if DW ... " << std::flush << scatter_if_test(WRAPPED) << std::endl;
	std::cout << "Gather DR ... " << std::flush << gather_test(RAW) << std::endl;
	std::cout << "Gather DW ... " << std::flush << gather_test(WRAPPED) << std::endl;
	std::cout << "Gather_if DR ... " << std::flush << gather_if_test(RAW) << std::endl;
	std::cout << "Gather_if DW ... " << std::flush << gather_if_test(WRAPPED) << std::endl;
	return 0;
}
