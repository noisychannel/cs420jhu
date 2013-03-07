package hw2;

public class CoinFlipMain {

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

		// Pre-calculate the number of iterations that will be assigned to each
		// thread
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
		System.out.println("Startup time "
				+ String.valueOf(intermediateRunTime) + " milliseconds");

		long totalHeadCounts = 0;

		// Await the completion of all threads
		for (int i = 0; i < numthreads; i++) {
			try {
				threads[i].join();
				totalHeadCounts = totalHeadCounts + coinFlips[i].getHeadCount();
			} catch (InterruptedException e) {
				System.out.println("Thread interrupted.  Exception: "
						+ e.toString() + " Message: " + e.getMessage());
				return;
			}
		}
		System.out.println("Heads = " + totalHeadCounts);
		System.out.println("Tails = " + (noOfIterations - totalHeadCounts));
		System.out.println("Elapsed time: "
				+ (System.currentTimeMillis() - runstart) + " milliseconds");
	}

	/**
	 * Calculates the number of coin flips that each thread will be responsible
	 * for.
	 * 
	 * @param noOfIterations
	 *            The number of coin flips
	 * @param numthreads
	 *            The number of threads
	 * @return
	 */
	private static long[] divideFlipsByThreads(int numthreads,
			long noOfIterations) {

		long[] flipCount = new long[numthreads];

		for (int i = 0; i < numthreads; i++) {
			flipCount[i] = noOfIterations / numthreads;
		}

		// Handle remainder
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
