#include "fix_gpu.cuh"
#include "kernel.cuh"

#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>
#include "thrust/async/reduce.h"
#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    }

int fix_image_gpu(Image& to_fix,cudaStream_t& stream)
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

    map_look_up(res,image_size);
 

    // #3 Histogram equalization

    // Histogram

    rmm::device_uvector<int> histo(256,stream);

    cudaMemsetAsync(histo.data(),0,histo.size()*sizeof(int),stream);

    cudaStreamSynchronize(stream);

    fill_histo(res,histo);
    
    cudaStreamSynchronize(stream);
    

    // Compute the inclusive sum scan of the histogram

    Scan_histo(histo);

    cudaStreamSynchronize(stream);
    // Find the first non-zero value in the cumulative histogram
    const int found = find_first_value(histo);

    // Apply the map transformation of the histogram equalization
    cudaStreamSynchronize(stream);

    last_mapping(res,histo,found); 

    cudaStreamSynchronize(stream);

    int ret = reduce(res);   
    
    cudaMemcpyAsync(to_fix.buffer,res.data(),image_size*sizeof(int),cudaMemcpyDeviceToHost,stream);

    cudaStreamSynchronize(stream);

    return ret;
}
