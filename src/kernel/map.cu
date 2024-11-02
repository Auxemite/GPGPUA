#include "kernel.cuh"

#include <raft/core/device_span.hpp>
#include <rmm/device_uvector.hpp>
#include <algorithm>


template <typename T>
__global__
void kernel_map_naif(raft::device_span<T> buffer)
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

template <typename T>
__global__
void kernel_last_mapping(raft::device_span<T> buffer,raft::device_span<T> histo,const int cdf)
{

    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;

    if(i>=buffer.size())
        return;
    buffer[i]=std::roundf(((histo[buffer[i]]-cdf)/static_cast<float>(buffer.size()-cdf))*255.0f);
}

template <typename T>
__global__
void kernel_map_lookUp(raft::device_span<T> buffer,raft::device_span<T> look)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x+tid;
    unsigned int mod = i%look.size();

    if(i>=buffer.size())
        return ;
    buffer[i]+=look[mod];
}


void map_classique(rmm::device_uvector<int>& buffer,const int image_size)
{
    int nb_block = (image_size+512-1)/512;
    kernel_map_naif<int><<<nb_block,512,0,buffer.stream()>>>(raft::device_span<int>(buffer.data(),image_size));
}


void last_mapping(rmm::device_uvector<int>& buffer,rmm::device_uvector<int>& histo,const int cdf)
{
    int nb_block = (buffer.size()+512-1)/512;
    kernel_last_mapping<int><<<nb_block,512,0,buffer.stream()>>>(raft::device_span<int>(buffer.data(),buffer.size()),raft::device_span<int>(histo.data(),histo.size()),cdf);
}


void map_look_up(rmm::device_uvector<int>& buffer,const int image_size)
{
    int nb_block = (image_size+512-1)/512;
    rmm::device_uvector<int> test(4,buffer.stream());
    test.set_element(0,1,buffer.stream());
    test.set_element(1,-5,buffer.stream());
    test.set_element(2,3,buffer.stream());
    test.set_element(3,-8,buffer.stream());
    kernel_map_lookUp<int><<<nb_block,512,0,buffer.stream()>>>(raft::device_span<int>(buffer.data(),image_size),raft::device_span<int>(test.data(),test.size()));
}


