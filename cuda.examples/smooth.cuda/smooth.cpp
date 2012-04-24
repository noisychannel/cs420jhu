#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>



/*
 *  Calculate the smoothed value for an individual point.
 *   This can be where arbitrarily complex algorithms may
 *   be implemented.
 */
float NNSmoother ( unsigned dim, float* buf )
{
  float value = 0;

  for ( unsigned j=0; j<dim; j++ )
  {
    for ( unsigned i=0; i<dim; i++ )
    {
      value += buf[j*dim + i];
    }
  }
  value /= dim*dim;
  return value;
}

/*
 *  Cut out an array of size (2*width+1)^2 
 *   from inField centered at x,y  and put it into outField
 */
void cut ( unsigned x, unsigned y, unsigned dim, unsigned width, float* inField, float* outField )
{
  unsigned cutdim = 2*width + 1; 

  for ( unsigned j=0; j<cutdim; j++ )
  {
    for ( unsigned i=0; i<cutdim; i++ )
    {
      outField[ cutdim*j + i ] = inField[ dim*(j+y-width) + x+i-width ];
    }
  }
  return;
}


/*
 *  Smooth a field of dimension dim, dim using a kernel of width x width
 *   pVector contains a 2-dim regular array of dim x dim.
 */
bool SmoothField ( unsigned dim, unsigned width, float* pFieldIn, float* pFieldOut ) 
{ 
  float* buf = (float *) malloc ((2*width+1)*(2*width+1)*sizeof(float) );
  memset (buf, 0, (2*width+1)*(2*width+1)*sizeof(float));

  for ( unsigned j=width; j<dim-width; j++ )
  {
    for ( unsigned i=width; i<dim-width; i++ )
    {
      cut ( i, j, dim, width, pFieldIn, buf);      
      pFieldOut[j*dim+i] = NNSmoother ( 2*width+1, buf );
    }
  }
  return true;
}


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

int main ()
{
  unsigned dimension = 2064;
  unsigned kernelwidth = 8; 

  struct timeval ta, tb;

  gettimeofday ( &ta, NULL );

  // Create the input field
  float *field = (float *) malloc ( dimension * dimension * sizeof(float));
  initField ( dimension, field );

  // Create the output field
  float *out = (float *) malloc ( dimension * dimension * sizeof(float));
  memset ( out, 0, dimension * dimension *sizeof(float));

  SmoothField ( dimension, kernelwidth, field, out );

  gettimeofday ( &tb, NULL );

  if ( ta.tv_usec < tb.tv_usec )
  {
    printf ("Elapsed total time (s/m): %d:%d\n", tb.tv_sec - ta.tv_sec, tb.tv_usec - ta.tv_usec );
  } else {
    printf ("Elapsed total time (s/m): %d:%d\n", tb.tv_sec - ta.tv_sec - 1, 1000000 - tb.tv_usec + ta.tv_usec );
  }



//  for ( unsigned j=0; j< dimension; j++ )
//  {
//    for ( unsigned i=0; i< dimension; i++ )
//    {
//      printf ("%4.4f, ", out[j*dimension + i]);
//    }
//    printf ("\n");
//  }
}

