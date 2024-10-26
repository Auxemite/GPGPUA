#include "fix_cpu.cuh"
#include "image.hh"

#include <string>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
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

void fix_image_gpu(Image& to_fix) {
    const int image_size = to_fix.width * to_fix.height;
    constexpr int garbage_val = -27;

    // Allocate device memory using thurst
    thrust::device_vector<int> d_buffer(to_fix.buffer, to_fix.buffer + to_fix.size());
    thrust::device_vector<int> d_predicate(to_fix.size(), 0);
    thrust::device_vector<int> d_histogram(256);
    print_log("Checkpoint 1");

    // #1 Compact - Build predicate vector
    thrust::transform(d_buffer.begin(), d_buffer.end(), d_predicate.begin(), [garbage_val] __device__(int val) {
        return val != garbage_val ? 1 : 0;
    });
    print_log("Checkpoint 2");

    // Compute the exclusive sum of the predicate (compact step)
    thrust::exclusive_scan(d_predicate.begin(), d_predicate.end(), d_predicate.begin());
    print_log("Checkpoint 3");

    // Scatter to the corresponding addresses
    // for (std::size_t i = 0; i < d_predicate.size(); ++i)
    //     if (d_buffer[i] != garbage_val)
    //         d_buffer[d_predicate[i]] = d_buffer[i];
    thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(to_fix.size()), [d_buffer = d_buffer.data(), d_predicate = d_predicate.data(), garbage_val] __device__(int idx) {
        if (d_buffer[idx] != garbage_val) {
            d_buffer[d_predicate[idx]] = d_buffer[idx];
        }
    });
    print_log("Checkpoint 4");
    
    // #2 Apply map to fix pixels
    const int block_size = 256;
    int grid_size = (image_size + block_size - 1) / block_size;
    apply_pixel_transformation<<<grid_size, block_size>>>(thrust::raw_pointer_cast(d_buffer.data()), image_size);
    print_log("Checkpoint 5");

    // #3 Histogram equalization
    // Calculate histogram
    histogram_kernel<<<grid_size, block_size>>>(thrust::raw_pointer_cast(d_buffer.data()), image_size, thrust::raw_pointer_cast(d_histogram.data()));
    print_log("Checkpoint 6");

    // Compute the inclusive sum scan of the histogram
    thrust::inclusive_scan(d_histogram.begin(), d_histogram.end(), d_histogram.begin());
    print_log("Checkpoint 7");

    // Find the first non-zero value in the cumulative histogram (on device)
    int cdf_min;
    auto first_non_zero = thrust::find_if(d_histogram.begin(), d_histogram.end(), [] __device__(int v) {
        return v != 0;
    });
    cudaMemcpy(&cdf_min, thrust::raw_pointer_cast(&(*first_non_zero)), sizeof(int), cudaMemcpyDeviceToHost);
    print_log("Checkpoint 8");

    // Apply histogram equalization transformation
    equalize_histogram<<<grid_size, block_size>>>(thrust::raw_pointer_cast(d_buffer.data()), image_size, thrust::raw_pointer_cast(d_histogram.data()), cdf_min);
    print_log("Checkpoint 9");

    // Copy the buffer back to host
    cudaMemcpy(to_fix.buffer, thrust::raw_pointer_cast(d_buffer.data()), sizeof(int) * to_fix.size(), cudaMemcpyDeviceToHost);
}