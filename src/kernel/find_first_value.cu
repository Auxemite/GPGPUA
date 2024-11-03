#include "kernel.cuh"

#include <raft/core/device_span.hpp>
#include <rmm/device_uvector.hpp>

#include <rmm/device_scalar.hpp>

template <typename T>
__global__
void kernel_first_val(raft::device_span<T> buffer,raft::device_span<T> res)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;

    if(buffer[i]==0)
        atomicAdd(&res[0],1);
    __syncthreads(); 

    if(tid==0)
        res[0]=buffer[res[0]];
}


int find_first_value(rmm::device_uvector<int>& buffer)
{
    rmm::device_scalar<int> res(0,buffer.stream());
    kernel_first_val<int><<<1,256,0,buffer.stream()>>>(raft::device_span<int>(buffer.data(),buffer.size()),raft::device_span<int>(res.data(),res.size()));
    return res.value(buffer.stream());
}
