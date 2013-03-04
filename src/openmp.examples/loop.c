#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "omp.h"

main () 
{
  int i;

  omp_set_num_threads ( 6 );

  #pragma omp parallel for
  for ( i=0; i<10; i++ )
  { 
    printf("OMP Thread# %d\n", omp_get_thread_num());
  }
}

