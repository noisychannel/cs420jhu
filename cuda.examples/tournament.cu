#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <cuda.h>

__global__ void TournamentKernel ( float* pVector, int stride )
{ 
  unsigned index = 2 * ( blockIdx.x * blockDim.x + threadIdx.x)  * stride ; 

  unsigned offset = threadIdx.x * stride;
  
  float tmpfloat;

  for ( unsigned i=1; i<=blockDim.x; i*=2 )
  {
    if  ( offset % (stride*i) == 0 ) 
    {
      if ( pVector[index] < pVector[index+stride*i] )
      {
        tmpfloat = pVector[index];
        pVector[index] = pVector[index+stride*i];
        pVector[index+stride*i] = tmpfloat;
      }
    }
     __syncthreads();
  }
} 


/*
 *
 *    N is the number of elements in the tournament.
 *       we are going to choose as flat a tournament as possible,
 *       which means grouping 512 elements at each lowest level.
 *       and start by limiting ourselves to two levels.
 *
 *    This subroutine only runs for power of 2 elements per block
 *       and the number of elements must either be 4 blkSize^2 or 
 *       8 blkSize^3.
 * 
 */
bool Tournament ( unsigned N, unsigned blkSize, float* pHostVector ) 
{ 
  unsigned stride = 1;
  unsigned BlockCount= N / 2 / blkSize;     
  unsigned VectorSize= N * sizeof(float); 

  float* pDeviceVector= 0; 

  assert (( N == 8 * blkSize * blkSize * blkSize ) || ( N == 4 * blkSize * blkSize ));
  assert ( (N & ( N-1))  == 0 );

  cudaMalloc((void**)&pDeviceVector, VectorSize); 
  cudaMemcpy(pDeviceVector, pHostVector, VectorSize, cudaMemcpyHostToDevice); 

//  OK let's start with a one-level invocation of a tournament
  while ( BlockCount > 0 )
  {
    printf ( "BlockCount %d, BlockSize %d, Stride %d\n", BlockCount, blkSize, stride );
    TournamentKernel <<<BlockCount,blkSize>>> ( pDeviceVector, stride ); 

    stride *= blkSize * 2;
    BlockCount /= blkSize *2;
  }

  cudaMemcpy(pHostVector, pDeviceVector, VectorSize, cudaMemcpyDeviceToHost); 

  return true;
}

int main ()
{
  int i;
  const int numels = 8 * 64;
  const int blocksize = 4;

  float hostvector[numels];

  // Initialize the input vectors 
  for ( i=0; i<numels; i++ )
  {
    hostvector[i] = i;
  }

  // Call the kernel
  Tournament ( numels, blocksize, hostvector ); 

  // Check the answer
  for ( i=0; i< numels; i++ )
  {
    printf ("Index/Value: %d/%4.4f\n", i, hostvector[i]);
  }
}

