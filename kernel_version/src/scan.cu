#include <cuda/atomic>
#include "kernel.cuh"

#include <rmm/device_scalar.hpp>
#include <raft/core/device_span.hpp>

template <typename T>
__global__
void kernel_scan_petit(raft::device_span<T> buffer)
{

    extern __shared__ T sdata[];
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int tid =threadIdx.x;
    if(i<buffer.size())
        sdata[tid]=buffer[i];
    else
        return;

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
    buffer[i]=sdata[tid];

}


void Scan_histo(rmm::device_uvector<int>& buffer)
{
    kernel_scan_petit<int><<<1,256,256*sizeof(int),buffer.stream()>>>(raft::device_span<int>(buffer.data(), buffer.size()));
}



template <typename T,int BLOCK_SIZE>
__global__
void kernel_DLB_scan(raft::device_span<T> buffer,raft::device_span<T> global_counter,raft::device_span<cuda::atomic<char,cuda::thread_scope_device>> flags,raft::device_span<int> somme_local,raft::device_span<int> somme_total)
{
    extern __shared__ T sdata[];
    if(threadIdx.x==0)
    {
        int id = atomicAdd(&global_counter[0],1);
        sdata[BLOCK_SIZE]=id;
        flags[id].store('X',cuda::memory_order_seq_cst);
    }
    __syncthreads();
    int blockID = sdata[BLOCK_SIZE];
    unsigned int tid =threadIdx.x;
    unsigned int i = blockID*blockDim.x+threadIdx.x;
    if(i<buffer.size())
        sdata[threadIdx.x]=buffer[i];
    else
        return;
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
    int oui = sdata[tid];
    int reduce_final = sdata[BLOCK_SIZE-1];
    if(blockID==0)
    { 
        buffer[i]=oui;
        if(threadIdx.x==0)
        {
            somme_total[blockID]=reduce_final;
            //__threadfence();
            flags[blockID].store('P',cuda::memory_order_seq_cst);
            flags[blockID].notify_all();
        }
    }
    else
    {
        if(threadIdx.x==0)
        {
            somme_local[blockID]=reduce_final;
            //__threadfence();
            flags[blockID].store('A',cuda::memory_order_seq_cst);
            flags[blockID].notify_all();
        }
        __syncthreads();
        if(threadIdx.x==0)
        {
            int to_add = 0;
            for(int id=blockID-1;id>=0;id--)
            {
                flags[id].wait('X',cuda::memory_order_seq_cst);  
                //__threadfence();
                int fl = flags[id].load(cuda::memory_order_seq_cst);
                if(fl=='A')
                {
                    to_add+=somme_local[id];
                }
                if(fl=='P')
                {
                    to_add+=somme_total[id];
                    break;
                }
            }
            //__threadfence();
            sdata[BLOCK_SIZE+1]=to_add;
        }
        __syncthreads();
        int whole_thing = sdata[BLOCK_SIZE+1];
        if(threadIdx.x==0)
        {
            somme_total[blockID]=reduce_final+whole_thing;
            //__threadfence();
            flags[blockID].store('P',cuda::memory_order_seq_cst);
            flags[blockID].notify_all();
        }
        __syncthreads();
        //__threadfence();
        buffer[i]=oui+whole_thing; 
    }
}

void DecoupledLookBack_Scan(rmm::device_uvector<int>& buffer)
{
	const int nb_block = (buffer.size()+512-1)/512;
    rmm::device_scalar<int> global_c(0,buffer.stream());
    rmm::device_uvector<cuda::atomic<char,cuda::thread_scope_device>> flags(nb_block,buffer.stream());
    rmm::device_uvector<int> somme_loc(nb_block,buffer.stream());
    rmm::device_uvector<int> somme_final(nb_block,buffer.stream());
    kernel_DLB_scan<int,512><<<nb_block,512,(512+2)*sizeof(int),buffer.stream()>>>(
        raft::device_span<int>(buffer.data(), buffer.size()),raft::device_span<int>(global_c.data(), global_c.size()),raft::device_span<cuda::atomic<char,cuda::thread_scope_device>>(flags.data(), flags.size()),raft::device_span<int>(somme_loc.data(), somme_loc.size()),raft::device_span<int>(somme_final.data(), somme_final.size()));
}
