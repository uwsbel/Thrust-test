Thrust-test
===========

A set of simple examples showing the use of Thrust algorithms using both raw and wrapped pointers.

The examples are organized in two folders:
* **old**: these examples use `cudaMalloc` and `cudaMemcpy`  (pre-CUDA 6)
* **new**: these examples use Unified Memory with `cudaMallocManaged` (CUDA 6 or newer)

_NOTE_: using Thrust with managed memory requires the latest development version Thrust v1.8, available from https://github.com/thrust/thrust (the CUDA Toolkit only provides Thrust v1.7).

### Unified Memory and Thrust

By default, Thrust relies on implicit algorithm dispatch, using tags associated with its vector containers. For example, the system tag for the iterators of `thrust::device_vector` is `thrust::cuda::tag`, so algorithms dispatched on such iterators will be parallelized in the CUDA system. This will not work with memory allocated through `cudaMallocManaged`. To prevent the need to introduce new vectors or to wrap existing managed memory simply to use a parallel algorithm, Thrust algorithms can be invoked with an explicitly specified execution policy. This approach is illustrated in the example below, where the array mA could also be directly passed, as is, to a host function or a CUDA kernel.

```c++
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
    double maximumGPU = thrust::reduce(thrust::cuda::par, mA, mA + ARRAY_SIZE, 0.0,      
                                       thrust::maximum<double>());
    cudaDeviceSynchronize();
    double maximumCPU = thrust::reduce(thrust::omp::par, mA, mA + ARRAY_SIZE, 0.0,    
                                       thrust::maximum<double>());
    std::cout << "GPU reduce: “ 
              << (std::fabs(maximumGPU ‐ ARRAY_SIZE) < 1e‐10 ? "Passed" : "Failed");
    std::cout << "CPU reduce: “ 
              << (std::fabs(maximumCPU ‐ ARRAY_SIZE) < 1e‐10 ? "Passed" : "Failed");
    cudaFree(mA);
    return 0;
}
```

With this model, the programmer specifies only the Thrust backend of interest (how the algorithm should be parallelized), without being concerned about the system being able to dereference the iterators provided to the algorithm (where the data "lives"). This is consistent with the simpler programming and memory management enabled by Unified Memory.
