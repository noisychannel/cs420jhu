package hw2;

import java.util.Random;

////////////////////////////////////////////////////////////////////////////////
//
//  class: CoinFlipping
//
//   Simple concurrent Java program demonstrating basic features of
//     thread creation
//     waiting on threads
//     synchronized blocks
//
////////////////////////////////////////////////////////////////////////////////

class CoinFlip implements Runnable {
	long noOfIterationsForThisThread;

	// Variables with counts for Heads and Tails
	// Instead of using static variables in class which are synchronized, 
	// I use private variables to avoid waits on resource locks.
	// No of tails is just numberOfFlips - headCount
	private long headCount = 0;

	// Run: overrides Runnabale.Run, thread entry point
	public void run() {
		Random random = new Random();
		for (int i = 0; i < this.noOfIterationsForThisThread; i++) {
			int toss = random.nextInt(2);
			if (toss == 1) {
				++this.headCount;
			}
		}
	}
	
	public long getHeadCount() {
		return this.headCount;
	}
	
	// Constructor: set number of iterations for this thread
	CoinFlip(long flipCount) {
		this.noOfIterationsForThisThread = flipCount;
	}

	public static void main(String[] args) {
		if (2 != args.length) {
			System.out.println("Usage: CoinFlip #threads #iterations");
			return;
		}
		
		long runstart;
		runstart = System.currentTimeMillis();

		// Get the number of threads we are going to run from the command line
		int numthreads = Integer.parseInt(args[0]);
		
		// Get the number of iterations for the coin flips
		long noOfIterations = Long.parseLong(args[1]);
		
		// Pre-calculate the number of iterations that will be assigned to each thread
		long[] flipCount = divideFlipsByThreads(numthreads, noOfIterations);

		// Array to hold references to thread objects
		Thread[] threads = new Thread[numthreads];
		
		// Array to hold runnable objects : CoinFlips
		CoinFlip[] coinFlips = new CoinFlip[numthreads];
		
		// create and start specified thread objects of class CoinFlip
		for (int i = 0; i < numthreads; i++) {
			CoinFlip coinFlipObject = new CoinFlip(flipCount[i]);
			coinFlips[i] = coinFlipObject;
			threads[i] = new Thread(coinFlipObject);
			threads[i].start();
		}
		
		long intermediateRunTime = System.currentTimeMillis() - runstart;
		System.out.println("Startup time " + String.valueOf(intermediateRunTime) + " milliseconds");
		
		long totalHeadCounts = 0;
		
		// Await the completion of all threads
		for (int i = 0; i < numthreads; i++) {
			try {
				threads[i].join();
				totalHeadCounts = totalHeadCounts + coinFlips[i].getHeadCount();
				System.out.println(coinFlips[i].getHeadCount());
			} catch (InterruptedException e) {
				System.out.println("Thread interrupted.  Exception: "
						+ e.toString() + " Message: " + e.getMessage());
				return;
			}
		}
		System.out.println("Heads = " + totalHeadCounts);
		System.out.println("Tails = " + (noOfIterations - totalHeadCounts));
		System.out.println("Elapsed time: " + (System.currentTimeMillis() - runstart) + " milliseconds");
	}

	/**
	 * Calculates the number of coin flips that each thread
	 * will be responsible for.
	 * @param noOfIterations The number of coin flips
	 * @param numthreads The number of threads
	 * @return 
	 */
	private static long[] divideFlipsByThreads(int numthreads, long noOfIterations) {

		long[] flipCount = new long[numthreads];
		
		for (int i = 0; i < numthreads; i++) {
			flipCount[i] = noOfIterations / numthreads;
		}
		
		//Handle remainder
		long remainder = noOfIterations % numthreads;
		int i = 0;
		
		while (remainder != 0) {
			flipCount[i] = flipCount[i] + 1;
			remainder = remainder - 1;
			i = i + 1;
			if (i > numthreads) {
				i = 0;
			}
		}
		
		return flipCount;
	}
}
