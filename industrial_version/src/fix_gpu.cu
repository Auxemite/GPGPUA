#include "fix_cpu.cuh"
#include "image.hh"

#include <string>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/async/scan.h>
#include <thrust/async/transform.h>
#include <thrust/async/remove.h>
// #include <raft/stats/histogram.cuh>
// #include <raft/matrix/matrix_view.hpp>
#include <rmm/device_uvector.hpp>
#include <cub/cub.cuh>

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

__global__ void apply_pixel_transformation(int* buffer, int image_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < image_size) {
        if (idx % 4 == 0)
            buffer[idx] += 1;
        else if (idx % 4 == 1)
            buffer[idx] -= 5;
        else if (idx % 4 == 2)
            buffer[idx] += 3;
        else if (idx % 4 == 3)
            buffer[idx] -= 8;
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
    const int image_size = to_fix.width * to_fix.height;
    
    // raft::resources handle;
    // Allocate device memory using thurst
    rmm::device_uvector<int> d_histogram(256);
    cudaMemsetAsync(d_histogram.data(), 0, sizeof(int) * 256, d_buffer.stream());
    cudaStreamSynchronize(d_buffer.stream());
    print_log("Checkpoint 1");

    // #1 Compact - Build predicate vector
    thrust::async::remove_if(thrust::cuda::par.on(d_buffer.stream()), d_buffer.begin(), d_buffer.end(), is_negate_27());
    cudaStreamSynchronize(d_buffer.stream());
    print_log("Checkpoint 2");
    
    // #2 Apply map to fix pixels
    const int block_size = 256;
    int grid_size = (image_size + block_size - 1) / block_size;
    apply_pixel_transformation<<<grid_size, block_size, 0, d_buffer.stream()>>>(d_buffer.data(), image_size);
    cudaStreamSynchronize(d_buffer.stream());
    print_log("Checkpoint 3");

    // #3 Histogram equalization
    // Calculate histogram
    // raft::device_matrix_view<const int, int, raft::col_major> data_view(d_buffer.data().get(), image_size, 1);
    // raft::device_matrix_view<int, int, raft::col_major> bins_view(d_histogram.data().get(), 256, 1);
    // raft::stats::histogram<int, int>(handle, raft::stats::HistType::BASIC, data_view, bins_view);
    histogram_kernel<<<grid_size, block_size, 0, d_buffer.stream()>>>(d_buffer.data(), image_size, d_histogram.data());
    cudaStreamSynchronize(d_buffer.stream());
    print_log("Checkpoint 4");

    // Compute the inclusive sum scan of the histogram
    thrust::async::inclusive_scan(thrust::cuda::par.on(d_buffer.stream()), d_histogram.begin(), d_histogram.end(), d_histogram.begin());
    cudaStreamSynchronize(d_buffer.stream());
    print_log("Checkpoint 5");

    // Find the first non-zero value in the cumulative histogram (on device)
    int cdf_min;
    auto first_non_zero = thrust::async::find_if(thrust::cuda::par.on(d_buffer.stream()), d_histogram.begin(), d_histogram.end(), [] __device__(int v) {
        return v != 0;
    });
    cudaMemcpyAsync(&cdf_min, thrust::raw_pointer_cast(&(*first_non_zero)), sizeof(int), cudaMemcpyDeviceToHost, d_buffer.stream());
    cudaStreamSynchronize(d_buffer.stream());
    print_log("Checkpoint 6");

    // Apply histogram equalization transformation
    equalize_histogram<<<grid_size, block_size, 0, d_buffer.stream()>>>(d_buffer.data(), image_size, d_histogram.data(), cdf_min);
    cudaStreamSynchronize(d_buffer.stream());
    print_log("Checkpoint 7");

    // Copy the buffer back to host
    cudaMemcpyAsync(to_fix.buffer, thrust::raw_pointer_cast(d_buffer.data()), sizeof(int) * to_fix.size(), cudaMemcpyDeviceToHost, d_buffer.stream());
}