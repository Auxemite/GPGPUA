#include "../image.hh"

#include <string>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/async/scan.h>
#include <thrust/async/transform.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <rmm/device_uvector.hpp>
#include <cub/cub.cuh>
#include <cub/device/device_histogram.cuh>

#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    }

void print_log(const std::string& message) {
    bool debug = false;
    if (debug)
        std::cout << message << std::endl;
}

struct HistogramEqualizationFunctor {
    int cdf_min;
    int image_size;

    HistogramEqualizationFunctor(int _cdf_min, int _image_size) : cdf_min(_cdf_min), image_size(_image_size) {}

    __device__ int operator()(int pixel, int cdf) const {
        return roundf(((cdf - cdf_min) / static_cast<float>(image_size - cdf_min)) * 255.0f);
    }
};

struct mod_index_functor {
    const int* values;

    mod_index_functor(const int* _values) : values(_values) {}

    __host__ __device__
    int operator()(const int& i) const {
        return values[i % 4];
    }
};

__global__ void apply_pixel_transformation(int* buffer, int image_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int values[] = {1, -5, 3, -8};
    if (idx < image_size) {
        buffer[idx] += values[idx % 4];
    }
}

__global__ void histogram_kernel(int* buffer, int image_size, int* histogram) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < image_size) {
        atomicAdd(&histogram[buffer[idx]], 1);
    }
}

__global__ void equalize_histogram(int* buffer, int image_size, int* histogram, int cdf_min) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < image_size) {
        float normalized = ((histogram[buffer[idx]] - cdf_min) / static_cast<float>(image_size - cdf_min)) * 255.0f;
        buffer[idx] = roundf(normalized);
    }
}

struct is_negate_27
{
  __host__ __device__
  bool operator()(const int x)
  {
    return x == -27;
  }
};

void fix_image_gpu(rmm::device_uvector<int>& d_buffer, const int image_size) {
    // raft::resources handle;
    // Allocate device memory using thurst
    rmm::device_uvector<int> d_histogram(256, d_buffer.stream());
    cudaMemsetAsync(d_histogram.data(), 0, sizeof(int) * 256, d_buffer.stream());
    cudaStreamSynchronize(d_buffer.stream());
    print_log("Checkpoint 1");

    // #1 Compact - Build predicate vector
    thrust::remove_if(thrust::cuda::par.on(d_buffer.stream()), d_buffer.begin(), d_buffer.end(), is_negate_27());
    cudaStreamSynchronize(d_buffer.stream());
    print_log("Checkpoint 2");
    
    // #2 Apply map to fix pixels
    const int block_size = 256;
    int grid_size = (image_size + block_size - 1) / block_size;
    // apply_pixel_transformation<<<grid_size, block_size, 0, d_buffer.stream()>>>(d_buffer.data(), image_size);

    thrust::device_vector<int> d_temp(image_size);
    thrust::sequence(d_temp.begin(), d_temp.end());
    const int values[] = {1, -5, 3, -8};
    mod_index_functor mod_index(values);
    thrust::transform(thrust::cuda::par.on(d_buffer.stream()), d_temp.begin(), d_temp.end(), d_temp.begin(), mod_index);
    thrust::transform(thrust::cuda::par.on(d_buffer.stream()), d_buffer.begin(), d_buffer.end(), d_temp.begin(), d_buffer.begin(), thrust::plus<int>());
    cudaStreamSynchronize(d_buffer.stream());
    print_log("Checkpoint 3");

    // #3 Histogram equalization
    // Calculate histogram
    int num_bins = 256;
    int min_val = 0;
    int max_val = 255;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Determine temporary device storage requirements
    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, thrust::raw_pointer_cast(d_buffer.data()), thrust::raw_pointer_cast(d_histogram.data()), num_bins, min_val, max_val + 1, image_size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Compute histogram with CUB
    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, thrust::raw_pointer_cast(d_buffer.data()), thrust::raw_pointer_cast(d_histogram.data()), num_bins, min_val, max_val + 1, image_size);

    // Clean up temporary storage
    cudaFree(d_temp_storage);

    // histogram_kernel<<<grid_size, block_size, 0, d_buffer.stream()>>>(d_buffer.data(), image_size, d_histogram.data());
    cudaStreamSynchronize(d_buffer.stream());
    print_log("Checkpoint 4");

    // Compute the inclusive sum scan of the histogram
    thrust::async::inclusive_scan(thrust::cuda::par.on(d_buffer.stream()), d_histogram.begin(), d_histogram.end(), d_histogram.begin());
    cudaStreamSynchronize(d_buffer.stream());
    print_log("Checkpoint 5");

    // Find the first non-zero value in the cumulative histogram (on device)
    int cdf_min;
    auto first_non_zero = thrust::find_if(thrust::cuda::par.on(d_buffer.stream()), d_histogram.begin(), d_histogram.end(), [] __device__(int v) {
        return v != 0;
    });
    cudaMemcpyAsync(&cdf_min, thrust::raw_pointer_cast(&(*first_non_zero)), sizeof(int), cudaMemcpyDeviceToHost, d_buffer.stream());
    cudaStreamSynchronize(d_buffer.stream());
    print_log("Checkpoint 6");

    // Apply histogram equalization transformation
    equalize_histogram<<<grid_size, block_size, 0, d_buffer.stream()>>>(d_buffer.data(), image_size, d_histogram.data(), cdf_min);
    // thrust::transform(d_buffer.begin(), d_buffer.end(), d_histogram.begin(), d_buffer.begin(), HistogramEqualizationFunctor(cdf_min, image_size));
    cudaStreamSynchronize(d_buffer.stream());
    print_log("Checkpoint 7");
}