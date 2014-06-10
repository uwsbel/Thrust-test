#include <iostream>
#include <cmath>

#include <thrust/version.h>
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
#include <thrust/inner_product.h>
#include <thrust/system/cuda/execution_policy.h>

const int ARRAY_SIZE = 1000;

enum Method {
  RAW,
  WRAPPED
};

bool inner_product_test(Method method)
{
  double *mA, *mB;

  cudaMallocManaged(&mA, sizeof(double) * ARRAY_SIZE);
  cudaMallocManaged(&mB, sizeof(double) * ARRAY_SIZE);

  for (int i = 0; i < ARRAY_SIZE; i++) {
    mA[i] = 1.0 * (i+1);
    mB[i] = 1.0 * (ARRAY_SIZE - i);
  }

  //// double inner_product = thrust::inner_product(thrust::cuda::par, mA, mA + ARRAY_SIZE, mB, 0.0, thrust::plus<double>(), thrust::multiplies<double>());
  double inner_product;
  switch (method) {
  case RAW:
    inner_product = thrust::inner_product(thrust::cuda::par, mA, mA + ARRAY_SIZE, mB, 0.0, thrust::plus<double>(), thrust::multiplies<double>());
    break;
  case WRAPPED:
    {
      thrust::device_ptr<double> wmA(mA), wmB(mB);
      inner_product = thrust::inner_product(thrust::cuda::par, wmA, wmA + ARRAY_SIZE, wmB, 0.0, thrust::plus<double>(), thrust::multiplies<double>());
      break;
    }
  default: break;
  }
  cudaDeviceSynchronize();

  double ref_inner_product = 0.0;

  for (int i = 0; i < ARRAY_SIZE; i++)
    ref_inner_product += mA[i] * mB[i];

  bool result = (fabs(inner_product - ref_inner_product) / fabs(ref_inner_product) < 1e-10);

  cudaFree(mA);
  cudaFree(mB);

  return result;
}

int main(int argc, char **argv) {
  int major = THRUST_MAJOR_VERSION;
  int minor = THRUST_MINOR_VERSION;
  std::cout << "Thrust v" << major << "." << minor << std::endl << std::endl;

  std::cout << "Inner_product DMR ... " << std::flush << inner_product_test(RAW) << std::endl;
  std::cout << "Inner_product DMW ... " << std::flush << inner_product_test(WRAPPED) << std::endl;

  return 0;
}
