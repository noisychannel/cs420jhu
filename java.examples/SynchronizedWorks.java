////////////////////////////////////////////////////////////////////////////////
//
//    $Id: SynchronizedWorks.java,v 1.2 2010-10-18 14:41:08 randal Exp $
//
//    Randal C. Burns
//    Department of Computer Science
//    Johns Hopkins University
//
//    $Source: /fshssl.home/randal/repository/class/420.2009/lectures/lec05/SynchronizedWorks.java,v $
//    $Date: 2010-10-18 14:41:08 $        
//    $Revision: 1.2 $
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//
//  class: SynchronizedWorks
//
//   Simple concurrent Java program demonstrating basic features of
//     thread creation
//     waiting on threads
//     synchronized blocks
//
////////////////////////////////////////////////////////////////////////////////

class SynchronizedWorks implements Runnable
{
  int thread_id;    // Variable containing specific id of this thread.
  
  // Create some variables for testing.
  static int sharedsynchvar = 0;
 
  // Run: overides Runnabale.Run, thread entry point
  public void run ()
  {
    for ( int i=0; i<10000000; i++ )
    {
      synchronized(SynchronizedWorks.class){sharedsynchvar++;}
    }
  }

  // Constructor: set thread id
  SynchronizedWorks ( int id ) 
  {
    this.thread_id = id;
  }

  public static void main ( String[] args )
  {
    if ( 1 != args.length ) 
    {
      System.out.println ("Usage: SynchronizedWorks #threads");
      return;
    } 

    // Get the number of threads we are going to run from the command line
    int numthreads = Integer.parseInt ( args[0] );

    // Array to hold references to thread objects
    Thread[] threads = new Thread[numthreads];

    // create and start specified thread objects of class SynchronizedWorks
    for ( int i=0; i<numthreads; i++ )
    {
      threads[i] = new Thread ( new SynchronizedWorks(i) );
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
    System.out.println("Shared synchronized variable = " + sharedsynchvar);
  }
}


////////////////////////////////////////////////////////////////////////////////
//
//  Revsion History 
//    
//  $Log: SynchronizedWorks.java,v $
//  Revision 1.2  2010-10-18 14:41:08  randal
//  Checkin.
//
//  Revision 1.1  2009/10/08 15:11:49  randal
//  Sync/checkin.  Move to rio.
//
//
////////////////////////////////////////////////////////////////////////////////
