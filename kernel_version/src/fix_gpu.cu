#include "fix_gpu.cuh"
#include "kernel.cuh"

#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <thrust/async/scan.h>

#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    }

void fix_image_gpu(Image& to_fix,cudaStream_t& stream)
{

    const int image_size = to_fix.width * to_fix.height;

    rmm::device_uvector<int> d_buffer(to_fix.size(),stream);
    
    cudaMemcpyAsync(d_buffer.data(),to_fix.buffer,to_fix.size()*sizeof(int),cudaMemcpyHostToDevice,stream); 
    
    cudaStreamSynchronize(stream);
    // #1 Compact

    // Build predicate vector
    rmm::device_uvector<int> predicate(d_buffer.size()+1,stream); 
    
    creation_mask(d_buffer,predicate);
    
    cudaStreamSynchronize(stream);
    // Compute the exclusive sum of the predicate

    //thrust::async::inclusive_scan(thrust::cuda::par.on(stream),predicate.begin(), predicate.end(), predicate.begin());
    DecoupledLookBack_Scan(predicate);
    
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Scatter to the corresponding addresses
    rmm::device_uvector<int> res(image_size,stream);
   

    scatter(d_buffer,predicate,res); 
    
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // #2 Apply map to fix pixels

    map_classique(res,image_size);

    cudaStreamSynchronize(stream);

    cudaMemcpyAsync(to_fix.buffer,res.data(),image_size*sizeof(int),cudaMemcpyDeviceToHost,stream);
    
    cudaStreamSynchronize(stream);
    /*
    // #3 Histogram equalization

    // Histogram

    std::array<int, 256> histo;
    histo.fill(0);
    for (int i = 0; i < image_size; ++i)
        ++histo[to_fix.buffer[i]];

    // Compute the inclusive sum scan of the histogram

    std::inclusive_scan(histo.begin(), histo.end(), histo.begin());

    // Find the first non-zero value in the cumulative histogram
    
    auto first_none_zero = std::find_if(histo.begin(), histo.end(), [](auto v) { return v != 0; });

    const int cdf_min = *first_none_zero;

    // Apply the map transformation of the histogram equalization

    std::transform(to_fix.buffer, to_fix.buffer + image_size, to_fix.buffer,
        [image_size, cdf_min, &histo](int pixel)
            {
                return std::roundf(((histo[pixel] - cdf_min) / static_cast<float>(image_size - cdf_min)) * 255.0f);
            }
    );*/
}
