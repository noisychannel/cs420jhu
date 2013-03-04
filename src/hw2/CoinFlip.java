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
	int noOfIterationsForThisThread;

	// Variables with counts for Heads and Tails
	static int headCount = 0;
	static int tailCount = 0;

	// Run: overrides Runnabale.Run, thread entry point
	public void run() {
		for (int i = 0; i < this.noOfIterationsForThisThread; i++) {
			Random random = new Random();
			int toss = random.nextInt(2);
			if (toss == 1) {
				synchronized (CoinFlip.class) {
					++tailCount;
				}
			} else {
				synchronized (CoinFlip.class) {
					++headCount;
				}
			}
		}
	}

	// Constructor: set thread id
	CoinFlip(int noOfIterationsForThisThread) {
		this.noOfIterationsForThisThread = noOfIterationsForThisThread;
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
		int noOfIterations = Integer.parseInt(args[1]);
		
		// Pre-calculate the number of iterations that will be assigned to each thread
		int[] flipCount = divideFlipsByThreads(numthreads, noOfIterations);

		// Array to hold references to thread objects
		Thread[] threads = new Thread[numthreads];
		
		// create and start specified thread objects of class CoinFlip
		for (int i = 0; i < numthreads; i++) {
			threads[i] = new Thread(new CoinFlip(flipCount[i]));
			threads[i].start();
		}
		
		long intermediateRunTime = System.currentTimeMillis() - runstart;
		System.out.println("Startup time " + String.valueOf(intermediateRunTime) + " milliseconds");
		
		// Await the completion of all threads
		for (int i = 0; i < numthreads; i++) {
			try {
				threads[i].join();
			} catch (InterruptedException e) {
				System.out.println("Thread interrupted.  Exception: "
						+ e.toString() + " Message: " + e.getMessage());
				return;
			}
		}
		System.out.println("Heads = " + headCount);
		System.out.println("Tails = " + tailCount);
		System.out.println("Elapsed time: " + (System.currentTimeMillis() - runstart) + " milliseconds");
	}

	/**
	 * Calculates the number of coin flips that each thread
	 * will be responsible for.
	 * @param noOfIterations The number of coin flips
	 * @param numthreads The number of threads
	 * @return 
	 */
	private static int[] divideFlipsByThreads(int numthreads, int noOfIterations) {
		int[] flipCount = new int[numthreads];
		
		for (int i = 0; i < numthreads; i++) {
			flipCount[i] = noOfIterations / numthreads;
		}
		
		//Handle remainder
		int remainder = noOfIterations % numthreads;
		int i = 0;
		
		while (remainder != 0) {
			flipCount[i] = flipCount[i] + 1;
			remainder = remainder - 1;
			i = i + 1;
		}
		
		return flipCount;
	}
}
