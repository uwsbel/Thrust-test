#cuda_include_directories(../)
#cuda_include_directories(../..)

# SOURCE_GROUP("MC64 Headers" FILES ${MC64_HEADERS})
# SOURCE_GROUP("MC64 CUDA Headers" FILES ${MC64_CUHEADERS})

cuda_add_executable(driver_sort sort.cu)
cuda_add_executable(driver_transformations transformations.cu)
cuda_add_executable(driver_fill_and_copy fill_and_copy.cu)
cuda_add_executable(driver_scatter_and_gather scatter_and_gather.cu)
