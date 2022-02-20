// Homework 1
// Color to Greyscale Conversion

//A common way to represent color images is known as RGBA - the color
//is specified by how much Red, Grean and Blue is in it.
//The 'A' stands for Alpha and is used for transparency, it will be
//ignored in this homework.

//Each channel Red, Blue, Green and Alpha is represented by one byte.
//Since we are using one byte for each color there are 256 different
//possible values for each color.  This means we use 4 bytes per pixel.

//Greyscale images are represented by a single intensity value per pixel
//which is one byte in size.

//To convert an image from color to grayscale one simple method is to
//set the intensity to the average of the RGB channels.  But we will
//use a more sophisticated method that takes into account how the eye 
//perceives color and weights the channels unequally.

//The eye responds most strongly to green followed by red and then blue.
//The NTSC (National Television System Committee) recommends the following
//formula for color to greyscale conversion:

//I = .299f * R + .587f * G + .114f * B

//Notice the trailing f's on the numbers which indicate that they are 
//single precision floating point constants and not double precision
//constants.

//You should fill in the kernel as well as set the block and grid sizes
//so that the entire image is processed.

#include "utils.h"
#include "stdio.h"

#define WARP_REDUCTION_32(value) \
    __syncthreads();\
    if ( threadIdx.x  < 16) value [ threadIdx.x ] += value [ threadIdx.x + 16];\
    if ( threadIdx.x  < 8)  value [ threadIdx.x ] += value [ threadIdx.x + 8];\
    if ( threadIdx.x  < 4)  value [ threadIdx.x ] += value [ threadIdx.x + 4];\
    if ( threadIdx.x  < 2)  value [ threadIdx.x ] += value [ threadIdx.x + 2];


#define WARP_REDUCTION_64(value)\
    __syncthreads();\
    if ( threadIdx.x  < 32) value [ threadIdx.x ] += value [ threadIdx.x + 32];\
    WARP_REDUCTION_32(value)


#define WARP_REDUCTION_128(value)\
    __syncthreads();\
    if ( threadIdx.x  < 64) value [ threadIdx.x ] += value [ threadIdx.x + 64];\
    WARP_REDUCTION_64(value)

#define WARP_REDUCTION_256(value)\
    __syncthreads();\
    if ( threadIdx.x  < 128) value [ threadIdx.x ] += value [ threadIdx.x + 128];\
    WARP_REDUCTION_128(value)



__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
  //TODO
  //Fill in the kernel to convert from color to greyscale
  //the mapping from components of a uchar4 to RGBA is:
  // .x -> R ; .y -> G ; .z -> B ; .w -> A
  //
  //The output (greyImage) at each pixel should be the result of
  //applying the formula: output = .299f * R + .587f * G + .114f * B;
  //Note: We will be ignoring the alpha channel for this conversion
  size_t curRow = blockIdx.x;
  size_t curCol = threadIdx.x;

  const uchar4* const cur = rgbaImage + curRow * numCols + curCol;
  unsigned char r = cur->x;
  unsigned char g = cur->y;
  unsigned char b = cur->z;
  
  greyImage[curRow*numCols+curCol] = (unsigned char)(r * 0.299 + g * 0.587 + b * 0.114);  

  //First create a mapping from the 2D block and grid locations
  //to an absolute 2D location in the image, then use that to
  //calculate a 1D offset
}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
  //You must fill in the correct sizes for the blockSize and gridSize
  //currently only one block with one thread is being launched
  const dim3 blockSize(numCols, 1, 1);  //TODO
  const dim3 gridSize(numRows, 1, 1);  //TODO
  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}

__global__ void vector_dotproduct_kernel(const float* a, const float* b, int len, int blen, float* result)
{
    __shared__ float value[256];
    int bstart = blen * blockIdx.x;
    int start = bstart + threadIdx.x;
    int end   = min(len, bstart + blen);

    float v = 0;
    for(int i = start; i < end; i += blockDim.x) v += (a[i] * b[i]);
    value[threadIdx.x] = v;

    // reduce to the first two values
    WARP_REDUCTION_256(value);

    // write back
    if ( threadIdx.x  == 0) result[blockIdx.x] = (value [0] + value[1]);
}


void test_cuda_product() {
    float *a;
    float *b;
    float *buf;
    cudaMalloc((void **) &a, 3000 * sizeof(float));
    cudaMalloc((void **) &b, 3000 * sizeof(float));
    cudaMalloc((void **) &buf, 32 * sizeof(float));

    float A[3000], B[3000],BUF[32];

    for (int i = 0; i < 3000; i++) A[i] = rand() * 1.0 / (float)RAND_MAX ;
    for (int i = 0; i < 3000; i++) B[i] = rand() * 1.0 / (float)RAND_MAX ;

    cudaMemcpy(a, A, 3000 * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy(b, B, 3000 * sizeof(float), cudaMemcpyHostToDevice );

    dim3 grid(32), block(256);

    vector_dotproduct_kernel<<<grid, block>>>(a, b, 3000, 256,  buf);

    cudaMemcpy(BUF, buf, 32 * sizeof(float), cudaMemcpyDeviceToHost );

    float gt = 0.0f;
    for (int i=0; i<3000; i++) gt += (A[i] * B[i]);
    std::cout << "gt :" << gt << std::endl;

    float sum = 0;
    for (int i=0; i<32; i++) sum += BUF[i];
    printf("results: %f\n", sum);
}