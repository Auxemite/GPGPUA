#include "kernel.cuh"

#include <raft/core/device_span.hpp>
#include <rmm/device_uvector.hpp>
#include <algorithm>

template <typename T,int GARBAGE_VALUE>
__global__
void kernel_scattering(raft::device_span<T> buffer,raft::device_span<T> scan)
{
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=scan.size()-1)
        return;
    
    if(buffer[i]!=GARBAGE_VALUE)
        buffer[predicate[i]]=buffer[i];

}



void scatter(rmm::device_uvector<int>& buffer,rmm::device_uvector<int>& predicate)
{
    int nb_block = ((predicate.size()-1)+512-1)/512;
    kernel_scattering<int,-27><<<nb_block,512,0,buffer.stream()>>>(raft::device_span<int>(buffer.data(),buffer.size()),raft::device_span<int>(predicate.data(),predicate.size()));
} 
