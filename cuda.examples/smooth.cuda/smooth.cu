/*******************************************************************************
*
*  kernelSmoother
*
*  This provides a CUDA implementation of a kernel smooother.
*   http://en.wikipedia.org/wiki/Kernel_smoother
*  The particular smoother in this file is a nearest neighbor smoother
*  in order to keep the code as simple to understand as possible.
*
*  This is implemeneted for 2-d square grids.
*
*  Parameters of note:
*    gridWidth -- size of the grid is gridWidth^2
*    blockWidth -- number of processors per block.
*        must be ((gridWidth-(kernelWidth-1)) % blockWidth == 0 
*        i.e. the smoothed regions must be of blocksize increments.
*    smoothWidth -- region around point x,y to smooth
*        must be odd, i.e. 2k+1 smooths box with corners (x-k,y-k) to (x+k,y+k)
*
*  The smoothed region is only defined for the interior that has the kernel
*   defined inside the boundary, e.g. for gridWidth=10, kernelWidth=2 the
*   region from 2,2 to 7,7 will be smoothed. 
*
********************************************************************************/

// TODO some testing.

/*******************************************************************************
*
*  CUDA concepts
*
*  This file shows how to use many features of CUDA:
*     2d grids
*     pitch allocation
*
********************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>

#include <cuda.h>

//
// SmoothGrid: structure to define geometry parameters.
//   set one of these up in main()
//
typedef struct
{
  unsigned int gridWidth;
  unsigned int blockWidth;
} SmoothGrid;

unsigned int smoothWidth;


/*------------------------------------------------------------------------------
* Name: NNSmoothKernel
* Action:  The CUDA kernel that implements kernel smoothing.
*             Yuck, that's two senses of kernel.
*-----------------------------------------------------------------------------*/
__global__ void NNSmoothKernel ( float* pFieldIn, float* pFieldOut, size_t pitch, CUDAGrid cg, unsigned int smoothwidth )
{ 
  // pitch is in bytes, figure out the number of elements for addressing
  unsigned pitchels = pitch/sizeof(float);

} 


/*------------------------------------------------------------------------------
* Name:  SmoothField
* Action:  Host entry point to kernel smoother
*-----------------------------------------------------------------------------*/
bool SmoothField ( float* pHostFieldIn, float *pHostFieldOut, CUDAGrid cg, unsigned int smoothWidth ) 
{ 
  float * pDeviceFieldIn = 0;
  float * pDeviceFieldOut = 0;

  size_t pitch, pitchout;

  struct timeval ta, tb, tc, td;

  // Check the grid dimensions and extract parameters.  See top description about restrictions
  assert ((( cg.kernelWidth -1 )%2) == 0 );     // Width is odd
  unsigned blockSize = cg.blockWidth * cg.blockWidth;  
  assert( ((cg.gridWidth-(cg.kernelWidth-1))*(cg.gridWidth-(cg.kernelWidth-1)) % blockSize) == 0 );

  gettimeofday ( &ta, NULL );

  // Place the data set on device memory
  cudaMallocPitch((void**)&pDeviceFieldIn, &pitch, cg.gridWidth*sizeof(float), cg.gridWidth ); 
  cudaMemcpy2D ( pDeviceFieldIn, pitch,
                 pHostFieldIn, cg.gridWidth*sizeof(float), cg.gridWidth*sizeof(float), cg.gridWidth,
                 cudaMemcpyHostToDevice); 

  // Allocate the output
  cudaMallocPitch((void**)&pDeviceFieldOut, &pitchout, cg.gridWidth*sizeof(float), cg.gridWidth ); 

  gettimeofday ( &tb, NULL );

  // Construct a 2d grid/block from the parameters in CUDAGrid
  const dim3 DimBlock .....TODO
  const dim3 DimGrid .....TODO

  // Invoke the kernel
  NNSmoothKernel <<<DimGrid,DimBlock>>> ( pDeviceFieldIn, pDeviceFieldOut, pitch, cg, smoothwidth ); 

  gettimeofday ( &tc, NULL );

  // Retrieve the results
  cudaMemcpy2D(pHostFieldOut, cg.gridWidth*sizeof(float), 
               pDeviceFieldOut, pitchout, cg.gridWidth*sizeof(float), cg.gridWidth,
               cudaMemcpyDeviceToHost); 

  gettimeofday ( &td, NULL );


  if ( ta.tv_usec < td.tv_usec )
  {
    printf ("Elapsed total time (s/m): %d:%d\n", td.tv_sec - ta.tv_sec, td.tv_usec - ta.tv_usec );
  } else {
    printf ("Elapsed total time (s/m): %d:%d\n", td.tv_sec - ta.tv_sec - 1, 1000000 - td.tv_usec + ta.tv_usec );
  }

  if ( tb.tv_usec < tc.tv_usec )
  {
    printf ("Elapsed kernel time (s/m): %d:%d\n", tc.tv_sec - tb.tv_sec, tc.tv_usec - tb.tv_usec );
  } else {
    printf ("Elapsed kernel time (s/m): %d:%d\n", tc.tv_sec - tb.tv_sec - 1, 1000000 - tc.tv_usec + tb.tv_usec );
  }

  return true;
}



/*------------------------------------------------------------------------------
* Name:  initField
* Action:  Initialize a field to predictable values.
*    This is a useful format for debugging, because values 
*    accumulate to their initial value.
*-----------------------------------------------------------------------------*/
void initField ( unsigned dim, float* pField )
{
  for ( unsigned j=0; j<dim; j++ )
  {
    for ( unsigned i=0; i<dim; i++ )
    {
      pField[j*dim+i] = j + i;
    }
  }
}


/*------------------------------------------------------------------------------
* Name:  main
* Action:  Entry point
*-----------------------------------------------------------------------------*/
int main ()
{

  // Define the grid
  CUDAGrid cg;
  cg.gridWidth = 4112;
  cg.blockWidth = 16;

  // Pick the width of the smoothing kernel
  smoothWidth = 17

  // Create the input field
  float *field = (float *) malloc ( cg.gridWidth * cg.gridWidth * sizeof(float));
  initField ( cg.gridWidth, field );

  // Create the output field
  float *out = (float *) malloc ( cg.gridWidth * cg.gridWidth * sizeof(float));

  // Call the kernel
  SmoothField ( field, out, cg, smoothWidth );

  // Print the output field (for debugging purposes.
/*  unsigned koffset = (cg.kernelWidth-1)/2;
  for ( unsigned j=0; j< cg.gridWidth; j++ )
  {
    for ( unsigned i=0; i< cg.gridWidth; i++ )
    {
      if ( ( i >= koffset ) && 
           ( j >= koffset ) &&
           ( i < ( cg.gridWidth - koffset )) &&
           ( j < ( cg.gridWidth - koffset )) )
      {
        printf ("%4.0f, ", out[j*cg.gridWidth + i]);
      }
      else
      {
        printf ("  na, ");
      }
    }  
    printf ("\n");
  }
*/
}

