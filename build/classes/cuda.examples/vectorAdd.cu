/*******************************************************************************
*
*  vectorAdd
*
*  Randal's reimplementation of the vector addition example code.  
*  This is the simplest case.  It lacks both
*    * padding
*    * bounds checking
*  So can only deal with vectors aligned with the ThreadBlocks.
*
********************************************************************************/

#include <stdio.h>
#include <cuda.h>

const unsigned BLOCKSIZE = 512; 

__global__ void VectorAdditionKernel ( 
    const float* pVectorA, 
    const float* pVectorB, 
    float* pVectorC  ) 
{ 
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
  pVectorC[i] = pVectorA[i] + pVectorB[i]; 
} 


bool VectorAddition ( 
    unsigned N, 
    const float* pHostVectorA, 
    const float* pHostVectorB, 
    float* pHostVectorC) 
{ 
  unsigned ThreadCount= N; 
  unsigned BlockCount= N / BLOCKSIZE; 
  unsigned VectorSize= ThreadCount* sizeof(float); 

  float* pDeviceVectorA= 0; 
  float* pDeviceVectorB= 0; 
  float* pDeviceVectorC= 0; 

  cudaMalloc((void**)&pDeviceVectorA, VectorSize); 
  cudaMalloc((void**)&pDeviceVectorB, VectorSize); 
  cudaMalloc((void**)&pDeviceVectorC, VectorSize); 

  cudaMemcpy(pDeviceVectorA, pHostVectorA, VectorSize, cudaMemcpyHostToDevice); 
  cudaMemcpy(pDeviceVectorB, pHostVectorB, VectorSize, cudaMemcpyHostToDevice); 

  VectorAdditionKernel <<<BlockCount,BLOCKSIZE>>> ( 
      pDeviceVectorA, 
      pDeviceVectorB, 
      pDeviceVectorC); 

  cudaMemcpy(pHostVectorC, pDeviceVectorC, VectorSize, cudaMemcpyDeviceToHost); 

  return true;
}

int main ()
{
  int i;

  float vecinput1[1024];
  float vecinput2[1024];
  float vecresult[1024];

  // Initialize the input vectors 
  for ( i=0; i<1024; i++ )
  {
    vecinput1[i] = i;
    vecinput2[i] = 1.0;
  }

  // Call the kernel
  VectorAddition ( 1024, vecinput1, vecinput2, vecresult ); 

  // Check the answer
  for ( i=0; i< 1024; i++ )
  {
    printf ("Index/Value: %d/%4.4f\n", i, vecresult[i]);
  }
}

