#include <iostream>
#include <cmath>
#include <thrust/reduce.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/omp/execution_policy.h>

const int ARRAY_SIZE = 1000;

int main(int argc, char **argv) {
  double* mA;
  cudaMallocManaged(&mA, ARRAY_SIZE * sizeof(double));
  
  thrust::sequence(mA, mA + ARRAY_SIZE, 1);

  double maximumGPU = thrust::reduce(thrust::cuda::par,
                                     mA, mA + ARRAY_SIZE,
                                     0.0, thrust::maximum<double>());
  cudaDeviceSynchronize();
  double maximumCPU = thrust::reduce(thrust::omp::par, 
                                     mA, mA + ARRAY_SIZE, 
                                     0.0, thrust::maximum<double>());
  
  std::cout << "GPU: " 
            << (std::fabs(maximumGPU - ARRAY_SIZE) < 1e-10 ? "OK" : "Failed")
            << std::endl;
  std::cout << "CPU: " 
            << (std::fabs(maximumCPU - ARRAY_SIZE) < 1e-10 ? "OK" : "Failed")
            << std::endl;
  
  cudaFree(mA);
  
  return 0;
}

 
