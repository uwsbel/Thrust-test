#cuda_include_directories(../)
#cuda_include_directories(../..)

# SOURCE_GROUP("MC64 Headers" FILES ${MC64_HEADERS})
# SOURCE_GROUP("MC64 CUDA Headers" FILES ${MC64_CUHEADERS})

cuda_add_executable(driver_sort_managed sort.cu)
cuda_add_executable(driver_transformations_managed transformations.cu)
cuda_add_executable(driver_fill_and_copy_managed fill_and_copy.cu)
cuda_add_executable(driver_scatter_and_gather_managed scatter_and_gather.cu)
cuda_add_executable(driver_reduce_managed reduce.cu)
cuda_add_executable(driver_transform_reduce_managed transform_reduce.cu)
