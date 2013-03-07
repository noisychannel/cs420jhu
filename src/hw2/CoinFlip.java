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

	private long noOfIterationsForThisThread;
	private Random random;

	// Variables with counts for Heads and Tails
	// Instead of using static variables in class which are synchronized,
	// I use private variables to avoid waits on resource locks.
	// No of tails is just numberOfFlips - headCount
	private long headCount = 0;

	// Constructor: set number of iterations for this thread
	CoinFlip(long flipCount) {
		this.noOfIterationsForThisThread = flipCount;
		this.random = new Random();
	}

	// Run: overrides Runnabale.Run, thread entry point
	public void run() {
		
		for (int i = 0; i < this.noOfIterationsForThisThread; i++) {
			int toss = this.random.nextInt(2);
			if (toss == 1) {
				++this.headCount;
			}
		}
	}

	public long getHeadCount() {
		return this.headCount;
	}

}
