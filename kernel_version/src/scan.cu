#include <cuda/atomic>


#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>
#include <raft/core/device_span.hpp>


template <typename T,int BLOCK_SIZE>
__global__
void kernel_DLB_scan(raft::device_span<T> buffer,raft::device_span<T> global_counter,raft::device_span<cuda::atomic<char,cuda::thread_scope_device>> flags,raft::device_span<T> somme_local,raft::device_span<T> somme_total)
{
    extern __shared__ T sdata[];
    if(threadIdx.x==0)
    {
        int id = atomicAdd(&global_counter[0],1);
        sdata[BLOCK_SIZE]=id;
        flags[id].store('X',cuda::memory_order_relaxed);
    }
    __syncthreads();
    int blockID = sdata[BLOCK_SIZE];
    unsigned int tid =threadIdx.x;
    unsigned int i = blockID*blockDim.x+threadIdx.x;
    if(i<buffer.size())
        sdata[threadIdx.x]=buffer[i];
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
    int reduce_final = sdata[BLOCK_SIZE-1];
    if(blockID==0)
    { 
        if(threadIdx.x==0)
        {
            somme_total[blockID]=reduce_final;
            flags[blockID].store('P',cuda::memory_order_relaxed);
            flags[blockID].notify_all();
        }
        buffer[i]=sdata[tid];
    }
    else
    {
        if(threadIdx.x==0)
        {
            somme_local[blockID] = reduce_final;
            __threadfence();
            flags[blockID].store('A',cuda::memory_order_relaxed);
            flags[blockID].notify_all();
        }
        __syncthreads();
        int to_add = 0;
        for(int id=blockID-1;id>=0;id--)
        {
            flags[id].wait('X',cuda::memory_order_relaxed);  
            __threadfence();
            int fl = flags[id].load(cuda::memory_order_relaxed);
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
        __syncthreads();
        if(threadIdx.x==0)
        {
            somme_total[blockID] = reduce_final+to_add;
            __threadfence();
            flags[blockID].store('P',cuda::memory_order_relaxed);
            flags[blockID].notify_all();
        }
        
        buffer[i]=to_add+sdata[tid]; 
    }
}

void DecoupledLookBack_Scan(rmm::device_uvector<int>& buffer)
{
	const int nb_block = (buffer.size()+512-1)/512;
    rmm::device_scalar<int> global_c(0,buffer.stream());
    rmm::device_uvector<cuda::atomic<char,cuda::thread_scope_device>> flags(nb_block,buffer.stream());
    rmm::device_uvector<int> somme_loc(nb_block,buffer.stream());
    rmm::device_uvector<int> somme_final(nb_block,buffer.stream());
    kernel_DLB_scan<int,512><<<nb_block,512,(512+1)*sizeof(int),buffer.stream()>>>(
        raft::device_span<int>(buffer.data(), buffer.size()),raft::device_span<int>(global_c.data(), global_c.size()),raft::device_span<cuda::atomic<char,cuda::thread_scope_device>>(flags.data(), flags.size()),raft::device_span<int>(somme_loc.data(), somme_loc.size()),raft::device_span<int>(somme_final.data(), somme_final.size()));
}
