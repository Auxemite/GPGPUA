#include "kernel.cuh"

#include <raft/core/device_span.hpp>
#include <rmm/device_uvector.hpp>

#include <rmm/device_scalar.hpp>

template <typename T>
__global__
void kernel_first_val(raft::device_span<T> buffer,raft::device_span<T> res)
{
    extern __shared__ T sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;

    sdata[tid]=buffer[i]; 
    __syncthreads();

    for(int s =1;s<blockDim.x;s*=2)
    {
        int pol=0;
        if(tid+s<blockDim.x)
	    {
		    pol=sdata[tid];
	    }
	    __syncthreads();
	    if(tid+s<blockDim.x)
	    {
            sdata[tid+s]+=pol;
	    }
        __syncthreads();
    }
    __syncthreads();

    if(tid==0)
        res[0]=sdata[0];
}


int find_first_value(rmm::device_uvector<int>& buffer)
{
    rmm::device_scalar<int> res(0,buffer.stream());
    kernel_first_val<int><<<1,256,256*sizeof(int),buffer.stream()>>>(raft::device_span<int>(buffer.data(),buffer.size()),raft::device_span<int>(res.data(),res.size()));

    return res.value(buffer.stream());
}
