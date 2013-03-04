package thread.javaExamples;

////////////////////////////////////////////////////////////////////////////////
//
//  class: VolatileWorks
//
//   Simple concurrent Java program demonstrating basic features of
//     thread creation
//     waiting on threads
//     the volatile keyword
//
////////////////////////////////////////////////////////////////////////////////

class VolatileWorks implements Runnable
{
  int thread_id;    // Variable containing specific id of this thread.
  
  // Create some variables for testing.
  static int sharedvar = 0;
  static volatile int sharedvolvar = 0;
 
  // Run: overides Runnabale.Run, thread entry point
  public void run ()
  {
    System.out.println ( thread_id + ": Running thread" );
    for ( int i=0; i<10000000; i++ )
    {
      sharedvar++;
    }
    System.out.println ( thread_id + ": Finished phase 1" );
    for ( int i=0; i<10000000; i++ )
    {
      sharedvolvar++;
    }
  }

  // Constructor: set thread id
  VolatileWorks ( int id ) 
  {
    this.thread_id = id;
  }

  public static void main ( String[] args )
  {
    if ( 1 != args.length ) 
    {
      System.out.println ("Usage: VolatileWorks #threads");
      return;
    } 

    // Get the number of threads we are going to run from the command line
    int numthreads = Integer.parseInt ( args[0] );

    // Array to hold references to thread objects
    Thread[] threads = new Thread[numthreads];

    // create and start specified thread objects of class VolatileWorks
    for ( int i=0; i<numthreads; i++ )
    {
      threads[i] = new Thread ( new VolatileWorks(i) );
      threads[i].start();
    }

    // Await the completion of all threads
    for ( int i=0; i<numthreads; i++ )
    {
      try
      {
        threads[i].join();
      }
      catch (InterruptedException e)
      {
         System.out.println("Thread interrupted.  Exception: " + e.toString() +
                           " Message: " + e.getMessage()) ;
        return;
      }
    }
    System.out.println("All threads complete.");
    System.out.println("Shared variable = " + sharedvar);
    System.out.println("Shared volatile variable = " + sharedvolvar);
  }
}


