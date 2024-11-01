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

template <typename T>
__global__
void kernel_fill_histo_meilleur(raft::device_span<T> buffer,raft::device_span<T> histo)
{
    __shared__ int sdata[256];
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;

    if(i>=buffer.size())
        return;

    for(int tid = threadIdx.x; tid<histo.size(); tid+=blockDim.x)
        sdata[tid]=0;
    __syncthreads();

    atomicAdd_block(&sdata[buffer[i]],1);
    __syncthreads();

    for(int tid=threadIdx.x; tid<histo.size(); tid+=blockDim.x)
        atomicAdd(&histo[tid],sdata[tid]);

}


void fill_histo(rmm::device_uvector<int>& buffer,rmm::device_uvector<int>& histo)
{
    int nb_block = (buffer.size()+256-1)/256;
    kernel_fill_histo_meilleur<int><<<nb_block,256,0,buffer.stream()>>>(raft::device_span<int>(buffer.data(),buffer.size()),raft::device_span<int>(histo.data(),histo.size()));
}
