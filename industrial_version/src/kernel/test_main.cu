#include "kernel.cuh"

#include <raft/core/handle.hpp>
#include <thrust/sequence.h>
#include <iostream>

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{

    raft::handle_t handle;
    cudaStream_t stream = handle.get_stream();
    rmm::device_uvector<int> d_vec(5897408, stream);
    thrust::sequence(thrust::cuda::par.on(stream),d_vec.begin(), d_vec.end(), 1);
    
    map_modulo(d_vec);
}
