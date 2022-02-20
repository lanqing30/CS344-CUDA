#include "utils.h"

// computeSum is used for two different things. v hacky
// templated by function to how to scan
template <typename T>
__global__
void hillis_steele_scan_kernel(const T * const d_arr, T * const d_out, const size_t N, 
                               const size_t binMask, const bool computeSum)
{
  extern __shared__ T temp[];

  int bdx = blockDim.x;
  int tid = threadIdx.x;
  int x = blockIdx.x * bdx + tid;
  
  if (x >= N)
    return;

  int in = 1;
  int out = 0;

  // need to make this exclusive scan
  temp[tid] = computeSum ? (((d_arr[x] & binMask) == 0) ? 1 : 0) : d_arr[x];
  __syncthreads();

  int s = 1;
  while (s < bdx) { // used to be N
    in = out;
    out = 1 - in;
    
    temp[out * bdx + tid] = temp[in * bdx + tid] + ((tid >= s) ? temp[in * bdx + tid - s] : 0);

    __syncthreads();
    s <<= 1;
  }

  d_out[x] = temp[out * bdx + tid];

  // fill in block sums
  if (computeSum) 
    if (threadIdx.x == blockDim.x - 1)
      d_out[N + blockIdx.x] = temp[out * bdx + tid];
}

template <typename T>
__global__
void add_sums_to_scan_kernel(T * const d_arr, const T * const d_block_sums, int N)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (x >= N)
    return;

  d_arr[x] += d_block_sums[blockIdx.x];
}

template <class T>
void hillis_steele_scan(const T * const d_input, T * const d_newPos, T * const d_block_sums,
    const size_t N, const size_t binMask, const size_t gridSize, const size_t blockSize)
{
    hillis_steele_scan_kernel<<<gridSize, blockSize, 2 * blockSize * sizeof(T)>>>(d_input, d_newPos, N, binMask, true);
    if (gridSize > 1) {
      if (gridSize > 1024) {
        // block sums : 1 2 3 ... 1024 1 2 3 ... 1024
        // form block scans over these to get
        //              1 3 6 ... <..> 1 3 6 ... <..>
        const size_t gridSize2 = round_up(gridSize, (size_t) 1024); 
        hillis_steele_scan_kernel<<<gridSize2, 1024, 2 * 1024 * sizeof(T)>>>(d_newPos + N, d_block_sums, gridSize, binMask, false);
        T *h_block_sums = (T *) malloc(gridSize * sizeof(T));
        checkCudaErrors(cudaMemcpy(h_block_sums, d_block_sums, gridSize * sizeof(T), cudaMemcpyDeviceToHost));
        T prev = 0;
        for (size_t i = 1024; i < gridSize; ++i) {
          if (i % 1024 == 0)
            prev = h_block_sums[i-1];
          h_block_sums[i] += prev;
        }
        checkCudaErrors(cudaMemcpy(d_block_sums, h_block_sums, gridSize * sizeof(T), cudaMemcpyHostToDevice));
        free(h_block_sums);
      } else {
        hillis_steele_scan_kernel<<<1, gridSize>>>(d_newPos + N, d_block_sums, gridSize, binMask, false);
      }
      add_sums_to_scan_kernel<<<gridSize-1, blockSize>>>(d_newPos + blockSize, d_block_sums, N - blockSize);
    }
}

// initialize specific instantiations of hillis_steele_scan
template void hillis_steele_scan<ull>(const ull * const, ull * const, ull * const,
        const size_t, const size_t, const size_t, const size_t);
