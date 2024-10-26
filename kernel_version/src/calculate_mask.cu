#include "kernel.cuh"

#include <raft/core/device_span.hpp>
#include <rmm/device_uvector.hpp>
#include <algorithm>

template <typename T,int GARBAGE_VALUE>
__global__
void kernel_mask(raft::device_span<T> buffer,raft::device_span<T> mask)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;

    if(i>=buffer.size())
        return;
    
    if(buffer[i]!=GARBAGE_VALUE)
        mask[i]=1;
}

void creation_mask(rmm::device_uvector<int>& buffer,rmm::device_uvector<int>& mask)
{
    int nb_block = (buffer.size()+128-1)/128;
    kernel_mask<int,-27><<<nb_block,128,0,buffer.stream()>>>(raft::device_span<int>(buffer.data(),buffer.size()),raft::device_span<int>(mask.data(),mask.size()))
}
