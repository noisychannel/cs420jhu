'''
Created on Mar 6, 2013

@author: "Gaurav Kumar"
'''

import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import collections


def getSpeedupData(logFileLocation):
    
    speedup = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    thread = 0;
    iteration = 0;
    
    logFile = open(logFileLocation)
    for line in logFile:
        if "Elapsed time" in line:
            runtime = re.findall('\d+',line)
            speedup[thread].append(int(runtime[0]))
            iteration = iteration + 1
            if iteration == 20:
                thread = thread + 1
                iteration = 0;
    
    minSpeedup = []
    for runTimes in speedup:
        minSpeedup.append(np.amin(runTimes))
    return minSpeedup


if __name__ == '__main__':
    
    speedup = []
    speedupFile = "speedup.log"
    
    speedup = getSpeedupData(speedupFile)
    print speedup
    
    plt.plot(range(1,17),[speedup[0]/x for x in speedup],marker="o",label="dualSmoothParallelYXFor")
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup (Ts/Tl)')
    plt.title('Speedup vs Number of Threads for dualSmoothParallelYXFor')
    plt.plot([1,16],[1,16],label="Linear (x=y)")
    plt.xlim(0,16)
    plt.ylim(0,16)
    plt.legend()
    
    plt.show()
        
        