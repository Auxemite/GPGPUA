#include "kernel.cuh"

#include <raft/core/device_span.hpp>

template <typename T>
__global__
void kernel_fill_histo(raft::device_span<T> buffer,raft::device_span<T> histo)
{
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;

    if(i>=buffer.size())
        return;

    atomicAdd(&histo[buffer[i]],1);
}

void fill_histo(rmm::device_uvector<int>& buffer,rmm::device_uvector<int>& histo)
{
    int nb_block = (buffer.size()+256-1)/256;
    kernel_fill_histo<int><<<nb_block,256,0,buffer.stream()>>>(raft::device_span<int>(buffer.data(),buffer.size()),raft::device_span<int>(histo.data(),histo.size()));
}
