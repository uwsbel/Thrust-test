/* The scattering on device is problematic. */

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

void scatter_test() {
	cout << "Scatter test ... " << flush;

	double *h_input, *d_input;
	double *h_output, *d_output;
	int    *h_map,  *d_map;

	h_input  = (double *)malloc (sizeof(double) * 10);
	h_output = (double *)malloc (sizeof(double) * 10);
	h_map    = (int *)   malloc (sizeof(int)    * 10);
	cudaMalloc(&d_input, sizeof(double) * 10);
	cudaMalloc(&d_output , sizeof(double) * 10);
	cudaMalloc(&d_map, sizeof(int) * 10);

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

	for (int i = 0; i < 10; i++)
		h_input[i] = 10.0 + (i + 1);

	cudaMemcpy(d_input, h_input, sizeof(double) * 10, cudaMemcpyHostToDevice);
	cudaMemcpy(d_output, h_output, sizeof(double) * 10, cudaMemcpyHostToDevice);
	cudaMemcpy(d_map,   h_map,   sizeof(int) * 10,    cudaMemcpyHostToDevice);

	//// thrust::scatter(thrust::cuda::par, d_input, d_input + 10, d_map, d_output);
	{
		thrust::device_ptr<double> input_begin(d_input), input_end(d_input + 10), output_begin(d_output);
		thrust::device_ptr<int>    map_begin(d_map);
		thrust::scatter(thrust::cuda::par, input_begin, input_end, map_begin, output_begin);
	}

	cudaMemcpy(h_output, d_output, sizeof(double) * 10, cudaMemcpyDeviceToHost);

	bool correct = true;
	for (int i = 0; i < 10; i++)
		if (h_output[h_map[i]] != h_input[i]) {
			correct = false;
			break;
		}

	cudaFree(d_map);
	cudaFree(d_input);
	cudaFree(d_output);
	free(h_map);
	free(h_output);
	free(h_input);

	if (correct)
		cout << "OK" << endl;
	else
		cout << "Failed" << endl;
}

void scatter_if_test() {
	cout << "Scatter if test ... " << flush;

	const double PRESET_VALUE = 10000.0;
	double *h_input, *d_input;
	double *h_output, *d_output;
	int    *h_map,  *d_map;
	bool   *h_stencil, *d_stencil;
	bool   h_output_visited[10] = {0};

	h_input  = (double *)malloc (sizeof(double) * 10);
	h_output = (double *)malloc (sizeof(double) * 10);
	h_map    = (int *)   malloc (sizeof(int)    * 10);
	h_stencil = (bool *) malloc (sizeof(bool)   * 10);
	cudaMalloc(&d_input, sizeof(double) * 10);
	cudaMalloc(&d_output , sizeof(double) * 10);
	cudaMalloc(&d_map, sizeof(int) * 10);
	cudaMalloc(&d_stencil, sizeof(bool) * 10);

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

	for (int i = 0; i < 10; i++) {
		h_input[i] = 10.0 + (i + 1);
		h_stencil[i] = ((i & 1) ? true : false);
		h_output[i] = PRESET_VALUE;
	}

	cudaMemcpy(d_input, h_input, sizeof(double) * 10, cudaMemcpyHostToDevice);
	cudaMemcpy(d_output, h_output, sizeof(double) * 10, cudaMemcpyHostToDevice);
	cudaMemcpy(d_stencil, h_stencil, sizeof(bool) * 10, cudaMemcpyHostToDevice);
	cudaMemcpy(d_map,   h_map,   sizeof(int) * 10,    cudaMemcpyHostToDevice);

	//// thrust::scatter_if(thrust::cuda::par, d_input, d_input + 10, d_map, d_stencil, d_output);
	{
		thrust::device_ptr<double> input_begin(d_input), input_end(d_input + 10), output_begin(d_output);
		thrust::device_ptr<int>    map_begin(d_map);
		thrust::device_ptr<bool>   stencil_begin(d_stencil);
		thrust::scatter_if(thrust::cuda::par, input_begin, input_end, map_begin, stencil_begin, output_begin, thrust::identity<bool>());
	}

	cudaMemcpy(h_output, d_output, sizeof(double) * 10, cudaMemcpyDeviceToHost);
		//// thrust::scatter_if(h_input, h_input + 10, h_map, h_stencil, h_output, thrust::identity<bool>());

	bool correct = true;
	for (int i = 0; i < 10; i++)
		if (h_stencil[i]) {
			h_output_visited[h_map[i]] = true;
			if (h_output[h_map[i]] != h_input[i]) {
				correct = false;
				break;
			}
		}

	if (correct) {
		for (int i = 0; i < 10; i++)
			if (!h_output_visited[i]) {
				if (h_output[i] != PRESET_VALUE) {
					correct = false;
					break;
				}
			}
	}

	cudaFree(d_map);
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_stencil);
	free(h_map);
	free(h_output);
	free(h_input);
	free(h_stencil);

	if (correct)
		cout << "OK" << endl;
	else
		cout << "Failed" << endl;

}

int main(int argc, char **argv) 
{
	scatter_test();
	scatter_if_test();
	return 0;
}
