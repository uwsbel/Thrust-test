CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(thrust_tests)
INCLUDE(SBELUtils.cmake)

SET(THRUST_DIR "" CACHE PATH "Location of the Thrust library")

enable_openmp_support()

enable_cuda_support()
MESSAGE(STATUS "Cuda version: ${CUDA_VERSION}")

cuda_include_directories(${THRUST_DIR})

ADD_SUBDIRECTORY(old)
IF(NOT (${CUDA_VERSION} VERSION_LESS "6.0"))
  ADD_SUBDIRECTORY(new)
ENDIF()
