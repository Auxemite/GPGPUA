#include "kernel.cuh"

#include <raft/core/device_span.hpp>
#include <rmm/device_uvector.hpp>
#include <algorithm>
template <typename T>
__global__
void kernel_map(raft::device_span<T> buffer)
{
    unsigned int tid = threadIdx.x;
    unsigned int mod = blockIdx.x%4;
    unsigned int i = int(blockIdx.x/4)*blockDim.x*4+mod+threadIdx.x*4;

    if(i>=buffer.size())
        return ;
    if(mod==0)
    {
        buffer[i]+=1;
    }
    else if(mod==1)
    {
        buffer[i]-=5;
    }
    else if(mod==2)
    {
        buffer[i]+=3;
    }
    else
    {
        buffer[i]-=8;
    }

}

template <typename T>
__global__
void kernel_map_2(raft::device_span<T> buffer)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x+tid;
    unsigned int mod = i%4;

    if(i>=buffer.size())
        return ;
    if(mod==0)
    {
        buffer[i]+=1;
    }
    else if(mod==1)
    {
        buffer[i]-=5;
    }
    else if(mod==2)
    {
        buffer[i]+=3;
    }
    else
    {
        buffer[i]-=8;
    }

}


void map_modulo(rmm::device_uvector<int>& buffer)
{
    unsigned int min_block =(buffer.size()+4-1)/4;
    int taille_block =(min_block+32-1)/32;
    int t = std::min(taille_block*32,1024);
    int nb_block = (buffer.size()+t-1)/t;
    if(nb_block%4!=0)
    {
        nb_block+=4-(nb_block%4);
    }
    kernel_map<int><<<nb_block,t,0,buffer.stream()>>>(raft::device_span<int>(buffer.data(),buffer.size()));
    //kernel_map_2<int><<<nb_block,t,0,buffer.stream()>>>(raft::device_span<int>(buffer.data(),buffer.size()));
}
