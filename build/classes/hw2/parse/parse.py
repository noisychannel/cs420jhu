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
            if iteration == 3:
                thread = thread + 1
                iteration = 0;
                
    minSpeedup = []
    for runTimes in speedup:
        minSpeedup.append(np.mean(runTimes))
    return minSpeedup

def getScaleupData(logFileLocation):
    
    scaleup = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    thread = 0;
    iteration = 0;
    
    logFile = open(logFileLocation)
    for line in logFile:
        if "Elapsed time" in line:
            runtime = re.findall('\d+',line)
            scaleup[thread].append(int(runtime[0]))
            iteration = iteration + 1
            if iteration == 3:
                thread = thread + 1
                iteration = 0;
    
    minScaleup = []
    for runTimes in scaleup:
        minScaleup.append(np.amin(runTimes))
    return minScaleup

def getDESSpeedupData(logFileLocation):
    
    speedup = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    thread = 0;
    iteration = 0;
    
    logFile = open(logFileLocation)
    for line in logFile:
        if "Completed search" in line:
            runtime = re.findall('\d+',line)
            speedup[thread].append(int(runtime[1]))
            iteration = iteration + 1
            if iteration == 5:
                thread = thread + 1
                iteration = 0;
    
    minSpeedup = []
    for runTimes in speedup:
        minSpeedup.append(np.mean(runTimes))
    return minSpeedup

def getDESScaleupData(logFileLocation):
    
    scaleup = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    thread = 0;
    iteration = 0;
    
    logFile = open(logFileLocation)
    for line in logFile:
        if "Completed search" in line:
            runtime = re.findall('\d+',line)
            scaleup[thread].append(int(runtime[1]))
            iteration = iteration + 1
            if iteration == 5:
                thread = thread + 1
                iteration = 0;
    
    minScaleup = []
    for runTimes in scaleup:
        minScaleup.append(np.mean(runTimes))
    return minScaleup


if __name__ == '__main__':
    
    speedupFile = "speedupCT.log"
    scaleupFile = "scaleupCT.log"
    
    DESspeedupFile = "speedupDES.log"
    DESscaleupFile = "scaleupDES.log"
    
    speedup = getSpeedupData(speedupFile)
    scaleup = getScaleupData(scaleupFile)
    DESspeedup = getDESSpeedupData(DESspeedupFile)
    DESscaleup = getDESScaleupData(DESscaleupFile)

#    for item in [float(speedup[0])/float(x) for x in speedup]:
#        print ("%.4f" % item)
#    plt.plot(range(1,17),[float(speedup[0])/float(x) for x in speedup],marker="o",label="CoinFlip")
#    plt.xlabel('Number of Threads')
#    plt.ylabel('Speedup (Ts/Tl)')
#    plt.title('Speedup vs Number of Threads for CoinFlip')
#    plt.plot([1,16],[1,16],label="Linear (x=y)")
#    plt.xlim(0,16)
#    plt.ylim(0,16)
#    plt.legend()
    
#    for item in [float(float(scaleup[0])/float(x)) for x in scaleup]:
#        print ("%.4f" % item)
#    plt.plot(range(1,17),[float(float(scaleup[0])/float(x)) for x in scaleup],marker="o",label="CoinFlip")
#    plt.xlabel('Number of Threads')
#    plt.ylabel('Scaleup (Ts/Tn)')
#    plt.title('Scaleup vs Number of Threads for CoinFlip')
#    plt.plot([0,16],[1,1],label="Linear (y=1)")
#    plt.xlim(0,16)
#    plt.ylim(0,2)
#    plt.legend()

#    for item in [float(float(DESspeedup[0])/float(x)) for x in DESspeedup]:
#        print ("%.4f" % item)
#    plt.plot(range(1,17),[float(float(DESspeedup[0])/float(x)) for x in DESspeedup],marker="o",label="BruteForceDES")
#    plt.xlabel('Number of Threads')
#    plt.ylabel('Speedup (Ts/Tl)')
#    plt.title('Speedup vs Number of Threads for BruteForceDES')
#    plt.plot([1,16],[1,16],label="Linear (x=y)")
#    plt.xlim(0,16)
#    plt.ylim(0,16)
#    plt.legend()
    
    for item in [float(float(DESscaleup[0])/float(x)) for x in DESscaleup]:
        print ("%.4f" % item)
    plt.plot(range(1,17),[float(float(DESscaleup[0])/float(x)) for x in DESscaleup],marker="o",label="BruteForceDES")
    plt.xlabel('Number of Threads')
    plt.ylabel('Scaleup (Ts/Tn)')
    plt.title('Scaleup vs Number of Threads for BruteForceDES')
    plt.plot([0,16],[1,1],label="Linear (y=1)")
    plt.xlim(0,16)
    plt.ylim(0,2)
    plt.legend()
    
    plt.show()
        
        