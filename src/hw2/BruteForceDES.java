package hw2;

import java.io.PrintStream;
import java.util.Random;

import javax.crypto.SealedObject;

public class BruteForceDES{

	public static void main(String[] args) {
		if (2 != args.length) {
			System.out
					.println("Usage: java BruteForceDES #threads #key_size_in_bits");
			return;
		}

		// create object to printf to the console
		PrintStream p = new PrintStream(System.out);

		// Get number of threads
		int numOfThreads = Integer.parseInt(args[0]);

		// Get the argument
		long keybits = Long.parseLong(args[1]);

		long maxkey = ~(0L);

		maxkey = maxkey >>> (64 - keybits);

		// Create a simple cipher
		SealedDES enccipher = new SealedDES();

		// Get a number between 0 and 2^64 - 1
		Random generator = new Random();
		long key = generator.nextLong();

		// Mask off the high bits so we get a short key
		key = key & maxkey;

		// Set up a key
		enccipher.setKey(key);

		// Generate a sample string
		String plainstr = "Johns Hopkins afraid of the big bad wolf?";

		// Encrypt
		SealedObject sldObj = enccipher.encrypt(plainstr);

		// Here ends the set-up. Pretending like we know nothing except sldObj,
		// discover what key was used to encrypt the message.

		// Get and store the current time -- for timing
		long runstart;
		runstart = System.currentTimeMillis();
		
		long[] searchesPerThread = divideSearchByThreads(numOfThreads, maxkey);

		// Array to hold references to thread objects
		Thread[] threads = new Thread[numOfThreads];

		// create and start specified thread objects of class CoinFlip
		for (int i = 0; i < numOfThreads; i++) {
			//Calculate search range
			long searchFrom;
			if (i == 0) {
				searchFrom = 0;
			}
			else {
				searchFrom = searchesPerThread[i-1];
			}
			
			long searchTo = searchesPerThread[i];
			
			threads[i] = new Thread(new SealedDES(searchFrom, searchTo, sldObj, i));
			threads[i].start();
		}

		// Await the completion of all threads
		for (int i = 0; i < numOfThreads; i++) {
			try {
				threads[i].join();
			} catch (InterruptedException e) {
				System.out.println("Thread interrupted.  Exception: "
						+ e.toString() + " Message: " + e.getMessage());
				return;
			}
		}

		// Output search time
		long elapsed = System.currentTimeMillis() - runstart;
		long keys = maxkey + 1;
		System.out.println("Completed search of " + keys + " keys at "
				+ elapsed + " milliseconds.");
	}
	
	/**
	 * Calculates the number of coin flips that each thread
	 * will be responsible for.
	 * @param noOfIterations The number of coin flips
	 * @param numthreads The number of threads
	 * @return 
	 */
	private static long[] divideSearchByThreads(int numthreads, long maxkey) {
		long[] flipCount = new long[numthreads];
		
		for (int i = 0; i < numthreads; i++) {
			flipCount[i] = maxkey / numthreads;
		}
		
		//Handle remainder
		long remainder = maxkey % numthreads;
		int i = 0;
		
		while (remainder != 0) {
			flipCount[i] = flipCount[i] + 1;
			remainder = remainder - 1;
			i = i + 1;
		}
		
		for (i = 0; i < numthreads; i++) {
			for (int j = 0; j < i; j++) {
				flipCount[i] = flipCount[i] + flipCount[j];
			}
		}
		
		return flipCount;
	}

}
