#include "kernel.cuh"
#include <raft/core/device_span.hpp>
#include <rmm/device_scalar.hpp>

template <typename T>
__inline__ __device__
void warp_reduce(T *shared_data, int tid) {
    if (tid < 32) shared_data[tid] += shared_data[tid + 32]; __syncwarp();
    if (tid < 16) shared_data[tid] += shared_data[tid + 16]; __syncwarp();
    if (tid < 8) shared_data[tid] += shared_data[tid + 8]; __syncwarp();
    if (tid < 4) shared_data[tid] += shared_data[tid + 4]; __syncwarp();
    if (tid < 2) shared_data[tid] += shared_data[tid + 2]; __syncwarp();
    if (tid < 1) shared_data[tid] += shared_data[tid + 1]; __syncwarp();
}

template <typename T>
__global__
void my_reduce(raft::device_span<const T> buffer, raft::device_span<T> total)
{
    extern __shared__ T shared_data[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    shared_data[tid] = 0;
    if (i < buffer.size())
      shared_data[tid] += buffer[i];
    if (i + blockDim.x < buffer.size())
      shared_data[tid] += buffer[i + blockDim.x];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s)
            shared_data[tid] += shared_data[tid + s];
        __syncthreads();
    }
    warp_reduce(shared_data, tid);

    if (tid == 0)
       total[0]=shared_data[0];
}

int reduce(rmm::device_uvector<int>& buffer)
{ 
    int nb_block = (buffer.size()+512-1)/512;
    rmm::device_scalar<int> global_c(0,buffer.stream());

    my_reduce<int><<<nb_block,512,512*sizeof(int),buffer.stream()>>>(raft::device_span<const int>(buffer.data(),buffer.size()),raft::device_span<int>(global_c.data(),global_c.size()));

    return global_c.value(buffer.stream());
} 
