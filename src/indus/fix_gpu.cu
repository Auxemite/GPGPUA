#include "../image.hh"

#include <string>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/async/scan.h>
#include <thrust/async/transform.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/find.h>
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

struct mod_index_functor {
    int *d_ptr;
    int *values;
    __host__ __device__
    int operator()(int& i) {
        d_ptr[i]+=values[i % 4];
    }
};

struct equalize {
    int *histo;
    const int cdf;
    const int image_size;
    __host__ __device__
    int operator()(const int& i) {
        return std::roundf(((histo[i]-cdf)/static_cast<float>(image_size-cdf))*255.0f);
    }
};


struct is_negate_27 {
  __host__ __device__ 
  bool operator()(const int& x)
  {
    return x == -27;
  }
};

void fix_image_gpu(rmm::device_uvector<int>& d_buffer, const int image_size) {
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
    rmm::device_uvector<int> values(4,d_buffer.stream());

    values.set_element(0, 1, d_buffer.stream()); 
    values.set_element(1, -5, d_buffer.stream()); 
    values.set_element(2, 3, d_buffer.stream()); 
    values.set_element(3, -8, d_buffer.stream()); 
    
    
    mod_index_functor op{d_buffer.data(),values.data()};


    std::uint8_t* d_temp_storage{};
    std::size_t temp_storage_bytes{};
    cub::DeviceFor::Bulk(d_temp_storage,temp_storage_bytes,image_size,op,d_buffer.stream());

    thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);

    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());
    
    cub::DeviceFor::Bulk(d_temp_storage,temp_storage_bytes,image_size,op,d_buffer.stream());
    
    cudaStreamSynchronize(d_buffer.stream());
    
    print_log("Checkpoint 3");

    // #3 Histogram equalization
    // Calculate histogram
    int num_bins = 257;
    int min_val = 0;
    int max_val = 255;
    void* d_temp_storage_2 = nullptr;
    size_t temp_storage_bytes_2 = 0;
    cub::DeviceHistogram::HistogramEven(d_temp_storage_2, temp_storage_bytes_2, d_buffer.data(), d_histogram.data(), num_bins, min_val, max_val + 1, image_size,d_buffer.stream());
    CUDA_CHECK(cudaStreamSynchronize(d_buffer.stream()));
    cudaMalloc(&d_temp_storage_2, temp_storage_bytes_2);
    cub::DeviceHistogram::HistogramEven(d_temp_storage_2, temp_storage_bytes_2, d_buffer.data(), d_histogram.data(), num_bins, min_val, max_val + 1, image_size,d_buffer.stream());

    cudaStreamSynchronize(d_buffer.stream());
    cudaFree(d_temp_storage_2);
    print_log("Checkpoint 4");

    // Compute the inclusive sum scan of the histogram
    thrust::async::inclusive_scan(thrust::cuda::par.on(d_buffer.stream()), d_histogram.begin(), d_histogram.end(), d_histogram.begin());
    CUDA_CHECK(cudaStreamSynchronize(d_buffer.stream()));
    print_log("Checkpoint 5");

    int cdf_min;
    // Find the first non-zero value in the cumulative histogram (on device)
    auto iter = thrust::find_if(thrust::cuda::par.on(d_buffer.stream()), d_histogram.begin(), d_histogram.end(), [] __host__ __device__(const int& v) {
        return v != 0;
    });
    CUDA_CHECK(cudaStreamSynchronize(d_buffer.stream()));
    cudaMemcpyAsync(&cdf_min,iter,sizeof(int),cudaMemcpyDeviceToHost,d_buffer.stream());
    
    cudaStreamSynchronize(d_buffer.stream());
    print_log("Checkpoint 6");

    // Apply histogram equalization transformation
    equalize op_2{d_histogram.data(),cdf_min,image_size};
    thrust::async::transform(thrust::cuda::par.on(d_buffer.stream()),d_buffer.begin(), d_buffer.end(), d_buffer.begin(),op_2);
    CUDA_CHECK(cudaStreamSynchronize(d_buffer.stream()));
    print_log("Checkpoint 7");
}
